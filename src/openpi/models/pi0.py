import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# make_attn_mask()：根据注意力可见性和 Token 有效性生成最终的 Attention Mask
#
# prefix_len：Prefix Token 的总长度
# 1. 有语言 Token：3 * patch_len + seq_len
# 2. 没有语言 Token：3 * patch_len
#
# suffix_len：Suffix Token 的总长度
# 1. 有状态 Token：1 + action_horizon
# 2. 没有状态 Token：action_horizon
#
# 参数：
# input_mask：[batch_size, prefix_len + suffix_len]
#   True：有效 Token
#   False：Padding Token
#
# ar_mask：[prefix_len + suffix_len]
#   True：当前 Token 开始新的 Attention 分组
#   False：当前 Token 与前一个 Token 属于同一 Attention 分组
#
# 返回：[batch_size, prefix_len + suffix_len, prefix_len + suffix_len]
#   True：允许 Query Token 关注 Key Token
#   False：禁止 Query Token 关注 Key Token

def make_attn_mask(
    input_mask,
    ar_mask,
):
    # 将 ar_mask 广播到与 input_mask 相同的形状
    # ar_mask：[prefix_len + suffix_len] -> [batch_size, prefix_len + suffix_len]
    ar_mask = jnp.broadcast_to(ar_mask, input_mask.shape)

    # jnp.cumsum(...): 布尔值会按照数值参与累积求和，True = 1，False = 0
    #
    # 沿 Token 序列维度进行累积求和，确定每个 Token 所属的 Attention 分组
    # 1. 累积值相同的 Token 属于同一组，可以相互关注
    # 2. 当前 Token 可以关注累积值小于或等于自己的 Token
    # 3. 当前 Token 不能关注累积值大于自己的 Token
    #
    # Eg:
    # ar_mask = [0, 0, 1, 0]
    # cumsum  = [0, 0, 1, 1]
    #
    # cumsum：[batch_size, prefix_len + suffix_len]
    cumsum = jnp.cumsum(ar_mask, axis=1)

    # 1. 注意力可见性（Attention Visibility）
    #    根据自回归分组规则，判断 Query Token 是否允许关注 Key Token
    #
    # cumsum[:, None, :]：[batch_size, 1, prefix_len + suffix_len]
    #   表示 Key Token 所属的 Attention 分组
    # cumsum[:, :, None]：[batch_size, prefix_len + suffix_len, 1]
    #   表示 Query Token 所属的 Attention 分组
    #
    # cumsum[:, None, :] <= cumsum[:, :, None]：
    # 1. 广播为 [batch_size, prefix_len + suffix_len, prefix_len + suffix_len]
    # 2. 逐位置比较，若 Key 分组 <= Query 分组则为 True，否则为 False
    #
    # Eg：取出某个 batch（用 1/0 表示 True/False）
    #     1. cumsum = [0, 0, 1, 1]
    #     2. attn_mask = [
    #         [1, 1, 0, 0],
    #         [1, 1, 0, 0],
    #         [1, 1, 1, 1],
    #         [1, 1, 1, 1]
    #     ]
    #     3. attn_mask[0, 1] == 1：0 号 Token 可以关注 1 号 Token
    #     4. attn_mask[0, 2] == 0：0 号 Token 不能关注 2 号 Token
    #
    # 行：Query Token（关注方）
    # 列：Key Token（被关注方）
    #
    # attn_mask：[batch_size, prefix_len + suffix_len, prefix_len + suffix_len]
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]

    # 2. Token 有效性（Token Validity）
    #    判断 Query Token 和 Key Token 是否均为有效 Token，而非 Padding Token
    #
    # input_mask[:, None, :]：[batch_size, 1, prefix_len + suffix_len]
    #   表示 Key Token 是否有效
    # input_mask[:, :, None]：[batch_size, prefix_len + suffix_len, 1]
    #   表示 Query Token 是否有效
    #
    # input_mask[:, None, :] * input_mask[:, :, None]：
    # 1. 广播为 [batch_size, prefix_len + suffix_len, prefix_len + suffix_len]
    # 2. 逐位置相乘，只有 Query 和 Key 均有效时结果才为 True
    #
    # Eg：取出某个 batch（用 1/0 表示 True/False）
    #     1. input_mask = [1, 1, 0, 1]
    #     2. valid_mask = [
    #         [1, 1, 0, 1],
    #         [1, 1, 0, 1],
    #         [0, 0, 0, 0],
    #         [1, 1, 0, 1]
    #     ]
    #     3. valid_mask[0, 1] == 1：0 号和 1 号 Token 均有效，具备相互关注的资格
    #     4. valid_mask[0, 2] == 0：2 号 Token 无效，0 号 Token 不能关注它
    #     5. valid_mask[2, :] == [0, 0, 0, 0]：2 号 Token 无效，不能作为有效 Query 关注任何 Token
    #
    # 行：Query Token（关注方）
    # 列：Key Token（被关注方）
    #
    # valid_mask：[batch_size, prefix_len + suffix_len, prefix_len + suffix_len]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]

    # 3. 综合判断：注意力可见性 + Token 有效性
    # jnp.logical_and()：Logical AND（逻辑与），逐位置进行逻辑与运算
    # 只有同时满足以下条件，最终结果才为 True：
    # 1. attn_mask == True：Attention 分组规则允许 Query 关注 Key
    # 2. valid_mask == True：Query 和 Key 均为有效 Token
    #
    # 返回：[batch_size, prefix_len + suffix_len, prefix_len + suffix_len]
    return jnp.logical_and(attn_mask, valid_mask)


# @at.typecheck：检查函数输入参数和返回值是否符合声明的类型及形状要求
@at.typecheck
# 将每个样本的 Flow Matching 时间 timestep（标量）转换为长度为 action_expert_width 的正弦-余弦时间嵌入向量 time_emb，让 Action Expert 能够利用当前时间信息
def posemb_sincos(
    # pos：[batch_size]，每个样本对应的 Flow Matching 时间 t
    # 在 π0 的训练阶段，pos 对应的就是 timestep，它是随机采样的，而且 Batch 中每个样本都有一个独立的时间值
    pos: at.Real[at.Array, " b"],

    # embedding_dim：时间嵌入向量的维度
    # 在 π0 中：embedding_dim = action_expert_width
    embedding_dim: int,

    # min_period：最小周期，例如 4e-3 = 0.004
    min_period: float,

    # max_period：最大周期，例如 4.0
    max_period: float,

# 返回：[batch_size, embedding_dim]，正弦-余弦时间嵌入向量
) -> at.Float[at.Array, "b {embedding_dim}"]:

    # 检查 embedding_dim 是否为偶数，因为一半维度用于 sin，另一半维度用于 cos
    # 例如：embedding_dim = 8，则 sin 和 cos 各占 4 维
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")


    # 生成归一化的周期分布位置
    #
    # embedding_dim // 2：需要生成的周期数量
    # jnp.linspace(0.0, 1.0, embedding_dim // 2)：
    #   在 [0, 1] 范围内生成 embedding_dim // 2 个等间隔数值
    #
    # 例如：embedding_dim = 8
    # fraction：[4] = [0.0, 0.3333, 0.6667, 1.0]
    #
    # fraction：[embedding_dim // 2]
    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)

    # 根据 fraction 生成从 min_period 到 max_period 的周期
    #
    # period = min_period * (max_period / min_period) ** fraction
    #
    # 当 fraction = 0：
    #   period = min_period
    #
    # 当 fraction = 1：
    #   period = max_period
    #
    # 中间周期按照几何比例分布，而不是线性分布
    #
    # 例如：embedding_dim = 8，min_period = 0.004，max_period = 4.0
    # fraction = [0.0, 1/3, 2/3, 1.0]
    # period   = [0.004, 0.04, 0.4, 4.0]
    #
    # period：[embedding_dim // 2]
    period = min_period * (max_period / min_period) ** fraction


    # 将每个时间 t 与不同周期对应的角频率相乘
    #
    # 角频率 omega = 2π / period
    #
    # pos：[batch_size]
    # period：[embedding_dim // 2]
    # omega = 2π / period：[embedding_dim // 2]
    #
    # jnp.einsum("i,j->ij", pos, 2π / period)：
    #   计算两个一维向量的外积
    #   每个样本的时间 t 都与全部角频率相乘
    #
    # 例如：batch_size = 2，embedding_dim = 8
    #
    # pos = [0.5, 1.0]
    # period = [0.004, 0.04, 0.4, 4.0]
    #
    # 1. 计算每个周期对应的角频率：
    #    2π / period = [500π, 50π, 5π, 0.5π]
    #    Shape：[embedding_dim // 2] = [4]
    #
    # 2. jnp.einsum("i,j->ij", pos, 2π / period)：
    #    将每个时间 t 与全部角频率相乘，计算两个一维向量的外积
    #
    #    sinusoid_input =
    #    [
    #        [250π, 25π, 2.5π, 0.25π],  # t = 0.5
    #        [500π, 50π, 5π,   0.5π ]   # t = 1.0
    #    ]
    #
    # sinusoid_input：[batch_size, embedding_dim // 2] = [2, 4]
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        # precision：Precision（计算精度），指定 jnp.einsum() 内部乘法和累加运算的精度
        # jax.lax.Precision.HIGHEST：请求使用当前计算平台支持的最高乘法精度，
        # 以减少浮点数计算误差，但不改变输入、输出的数据类型（dtype）
        precision=jax.lax.Precision.HIGHEST,
    )

    # 分别计算 sin 和 cos，再沿最后一个维度拼接
    #
    # jnp.sin(sinusoid_input)：
    #   [batch_size, embedding_dim // 2]
    #
    # jnp.cos(sinusoid_input)：
    #   [batch_size, embedding_dim // 2]
    #
    # jnp.concatenate(..., axis=-1)：
    #   沿最后一个维度拼接 sin 和 cos
    #
    # 例如：embedding_dim = 8
    #   前 4 维：sin(t * 2π / period)
    #   后 4 维：cos(t * 2π / period)
    #
    # 返回 time_emb：[batch_size, embedding_dim]
    # 在 π0 中：[batch_size, action_expert_width]
    return jnp.concatenate(
        [jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)],
        axis=-1,
    )

# PI0 模型
class Pi0(_model.BaseModel):

    def __init__(
        self, 
        # PI0 配置参数
        config: pi0_config.Pi0Config, 
        # 随机数密钥（Random Key）：用于控制随机数的生成，使随机操作具有可复现性
        #
        # 随机数密钥管理器（Random Key Manager）：用于管理不同用途的随机数密钥，并按需生成新的随机数密钥
        #
        # 根据 JAX 的随机数密钥 rng，随机数密钥管理器 rngs（nnx.Rngs），用于管理不同用途的随机数密钥，并按需生成新的随机数密钥
        rngs: nnx.Rngs
    ):

        # 调用父类的初始化函数
        super().__init__(
            # 动作维度：一个动作需要多少个部分共同完成
            config.action_dim, # 32
            # 动作序列长度：一次预测产生的动作数量
            config.action_horizon, # 50
            # 最大 token 数：语言提示可最多包含的 token 数
            config.max_token_len # 48
        )

        #  是否启用 PI05 模型（Bool)
        self.pi05 = config.pi05 # False

        # 根据 Pi0Config 中指定的模型配置名称，获取 PaliGemma 主干模型的配置
        # config.paligemma_variant == "gemma_2b"
        # paligemma_config == Config(
        #     width=2048,      # 隐藏层维度
        #     depth=18,        # Transformer 层数
        #     mlp_dim=16_384,  # MLP 中间层维度
        #     num_heads=8,     # Query（查询向量）的注意力头数量
        #     num_kv_heads=1,  # Key（键向量）和 Value（值向量）的注意力头数量
        #     head_dim=256,    # 每个注意力头的特征维度
        # )
        paligemma_config = _gemma.get_config(config.paligemma_variant)

        # 根据 Pi0Config 中指定的模型配置名称，获取 Action Expert 动作专家的配置
        # config.action_expert_variant == "gemma_300m"
        # action_expert_config == Config(
        #     width=1024,      # 隐藏层维度
        #     depth=18,        # Transformer 层数
        #     mlp_dim=4096,    # MLP 中间层维度
        #     num_heads=8,     # Query（查询向量）的注意力头数量
        #     num_kv_heads=1,  # Key（键向量）和 Value（值向量）的注意力头数量
        #     head_dim=256,    # 每个注意力头的特征维度
        # )
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # 创建 π0 的 Gemma Transformer 模型，并将其包装为 Flax NNX 兼容模块。
        # 其中包含两个 Expert：
        #   1. Expert 0：PaliGemma（VLM），负责处理视觉、语言等多模态特征。
        #   2. Expert 1：Action Expert（动作专家），负责处理机器人状态和带噪动作等特征。
        # 两个 Expert 使用各自的模型参数，通过联合 Attention 实现信息交互。
        # 
        # ToNNX 是 OpenPI 使用的兼容桥接工具，用于将 Flax Linen 模块包装为可以通过 Flax NNX 风格接口使用的模块
        llm = nnx_bridge.ToNNX(
            # _gemma.Module：基于 Flax Linen 实现的多 Expert Transformer 模型
            _gemma.Module(
                # 各 Expert 的 配置实例
                configs=[
                    # VLM 的 Config 配置实例
                    paligemma_config, 
                    # Action Expert 的 Config 配置实例
                    action_expert_config
                ],
                # 词嵌入结果的指定类型
                embed_dtype=config.dtype, # "bfloat16"
                adarms=config.pi05, # False
            )
        )

        # lazy_init()：通过虚拟输入执行模型的参数初始化，不进行实际训练
        llm.lazy_init(
            # 用于管理随机数密钥的对象 rngs（nnx.Rngs），用于初始化模型参数
            # lazy_init() 所需要的参数
            rngs=rngs, 
            # 指定调用 _gemma.Module 中自定义的 init() 方法
            # lazy_init() 所需要的参数
            method="init", 
            # VLM 和 Action Expert 是否启用 Adaptive RMSNorm（
            # _gemma.Module.init() 所需要的参数，由 lazy_init() 转发给 Module.init() 函数
            use_adarms=[False, True] if config.pi05 else [False, False] # [False, False]
        )

        # 创建视觉编码器 img
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                # 以下参数被当成字典进行传入
                # kw = {
                #     "pool_type": "none",
                #     "scan": True,
                #     "dtype_mm": "bfloat16",
                # }
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )

        # config.fake_obs()
        # observation = Observation(
        #     images={                                  # 图像：[batch_size, height, width, channels]
        #         "base_0_rgb": ...,                    # [1, 224, 224, 3]，机器人基座视角
        #         "left_wrist_0_rgb": ...,              # [1, 224, 224, 3]，左腕部摄像头视角
        #         "right_wrist_0_rgb": ...,             # [1, 224, 224, 3]，右腕部摄像头视角
        #     },
        #     image_masks={                             # 图像掩码：[batch_size]
        #         "base_0_rgb": ...,                    # [1]，基座视角图像是否有效
        #         "left_wrist_0_rgb": ...,              # [1]，左腕部图像是否有效
        #         "right_wrist_0_rgb": ...,             # [1]，右腕部图像是否有效
        #     },
        #     state=...,                                # 机器人状态：[batch_size, state_dimension]，形状为 [1, 32]
        #     tokenized_prompt=...,                     # 语言 Token ID：[batch_size, seq_len]，形状为 [1, 48]
        #     tokenized_prompt_mask=...,                # 语言 Token 掩码：[batch_size, seq_len]，形状为 [1, 48]
        #     token_ar_mask=None,                       # 自回归掩码：[batch_size, seq_len]，π0-FAST 专用
        #     token_loss_mask=None,                     # 损失掩码：[batch_size, seq_len]，π0-FAST 专用
        # )

        # 初始化 视觉编码器 image 的初始权重
        # config.fake_obs().images.values() 得到一个 dict_values 类型的可迭代实例，其中每个元素都是 image 这个字典中键值对的值
        # iter(dict_values()): 将 dict_values 封装成一个迭代器，可以通过 next 依次取出迭代器中的元素
        # next(iter(dict_values())): 取出 dict_values 的第一个元素，即 "base_0_rgb" 所对应的 value
        # next(iter(dict_values())): [batch_size(1), heigh(224), width(224), channels(3)]
        img.lazy_init(
            # 提供一张虚拟图像作为输入，用于推断输入形状并初始化参数
            next(iter(config.fake_obs().images.values())), 
            # 指定非训练模式
            train=False, 
            # 提供初始化所需的随机数生成器
            rngs=rngs
        )

        # nnx.Dict 是 Flax NNX 提供的字典式容器，可以把多个模块组织在一起，并通过键访问它们
        # self.PaliGemma
        # │
        # ├── "llm" ──> llm
        # │             └── Gemma Transformer
        # │                 ├── PaliGemma
        # │                 └── Action Expert
        # │
        # └── "img" ──> img
        #               └── SigLIP 视觉编码器
        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        # 创建动作输入投影层：将动作从 action_dim 维映射到 Action Expert 的隐藏层维度
        self.action_in_proj = nnx.Linear(
            in_features=config.action_dim,                  # 输入维度：32
            out_features=action_expert_config.width,        # 输出维度：1024
            rngs=rngs,                                      # 随机数生成器，用于初始化线性层参数
        )

        if config.pi05:  # 是否使用 Pi0.5 模型；当前为 False
            # 创建时间嵌入 MLP 的输入层
            self.time_mlp_in = nnx.Linear(
                in_features=action_expert_config.width,     # 输入维度：1024
                out_features=action_expert_config.width,    # 输出维度：1024
                rngs=rngs,                                  # 随机数生成器，用于初始化线性层参数
            )

            # 创建时间嵌入 MLP 的输出层
            self.time_mlp_out = nnx.Linear(
                in_features=action_expert_config.width,     # 输入维度：1024
                out_features=action_expert_config.width,    # 输出维度：1024
                rngs=rngs,                                  # 随机数生成器，用于初始化线性层参数
            )

        else:
            # 创建机器人状态投影层：将机器人状态映射到 Action Expert 的隐藏特征空间
            self.state_proj = nnx.Linear(
                in_features=config.action_dim,              # 输入维度：32
                out_features=action_expert_config.width,    # 输出维度：1024
                rngs=rngs,                                  # 随机数生成器，用于初始化线性层参数
            )

            # 创建动作与时间嵌入融合后的 MLP 输入层
            self.action_time_mlp_in = nnx.Linear(
                in_features=2 * action_expert_config.width, # 输入维度：2048，动作特征与时间特征拼接
                out_features=action_expert_config.width,    # 输出维度：1024
                rngs=rngs,                                  # 随机数生成器，用于初始化线性层参数
            )

            # 创建动作与时间嵌入融合后的 MLP 输出层
            self.action_time_mlp_out = nnx.Linear(
                in_features=action_expert_config.width,     # 输入维度：1024
                out_features=action_expert_config.width,    # 输出维度：1024
                rngs=rngs,                                  # 随机数生成器，用于初始化线性层参数
            )

        # 创建动作输出投影层：将 Action Expert 的隐藏特征映射回动作维度
        self.action_out_proj = nnx.Linear(
            in_features=action_expert_config.width,         # 输入维度：1024
            out_features=config.action_dim,                 # 输出维度：32
            rngs=rngs,                                      # 随机数生成器，用于初始化线性层参数
        )

        # 确定性模式标志：默认设为 True，表示受该标志控制的 Dropout 不进行随机丢弃
        # model.train() 和 model.eval() 会自动更新该属性
        self.deterministic = True

    # embed_prefix()：将摄像头图像和语言 Token 编码为前缀特征序列，
    # 并生成对应的有效性掩码 input_mask 和注意力分组标记 ar_mask。
    #
    # 输入：
    # obs：机器人观测数据，包含图像、图像掩码、语言 Token 及其掩码。
    #
    # 返回：
    # tokens：[batch_size, prefix_len, paligemma_width]，前缀特征序列。
    # input_mask：[batch_size, prefix_len]，输入有效性掩码。
    # ar_mask：[prefix_len]，注意力分组标记。

    # @at.typecheck：检查函数输入参数和返回值是否符合声明的类型及形状要求
    @at.typecheck
    def embed_prefix(
        self,
        obs: _model.Observation,  # 机器人观测数据：图像、图像掩码、语言 Token 及其掩码
    ) -> tuple[
        at.Float[at.Array, "b s emb"],  # tokens：[batch_size, prefix_len, paligemma_width]
        at.Bool[at.Array, "b s"],       # input_mask：[batch_size, prefix_len]
        at.Bool[at.Array, " s"],        # ar_mask：[prefix_len]
    ]:
        input_mask = []  # 保存图像 Token 和语言 Token 的有效性掩码
        ar_mask = []     # 保存各个 Token 的注意力分组标记
        tokens = []      # 保存图像 Token 和语言 Token 的特征

        # ar_mask 的含义：
        # True：开启一个新的注意力分组
        # False：延续当前注意力分组
        # 后续根据 ar_mask 构造实际的注意力掩码，决定不同 Token 之间的注意力关系

        # 1. 编码所有摄像头的图像
        for name in obs.images:

            # 将图像输入 SigLIP 视觉编码器，提取图像 Token 特征
            # obs.images[name]：[batch_size, height, width, channels]
            # patch_len = patch_rows * patch_cols：单张图像的 Patch 总数
            # image_tokens：[batch_size, patch_len, paligemma_width]
            # train=False：以非训练模式运行视觉编码器
            image_tokens, _ = self.PaliGemma.img(
                # 未指定 method 参数，默认调用 img 模块的 __call__() 方法
                # 其余参数均作为 __call__() 方法的参数传入
                obs.images[name],
                train=False,
            )

            # 1.1 将当前摄像头的图像 Token 添加到 tokens 列表
            tokens.append(image_tokens)

            # 例如，处理完 "base_0_rgb" 后：
            # tokens = [
            #     image_tokens：[batch_size, patch_len, paligemma_width]  # "base_0_rgb"
            # ]

            # 处理完三路摄像头后：
            # tokens = [
            #     image_tokens：[batch_size, patch_len, paligemma_width]  # "base_0_rgb"
            #     image_tokens：[batch_size, patch_len, paligemma_width]  # "left_wrist_0_rgb"
            #     image_tokens：[batch_size, patch_len, paligemma_width]  # "right_wrist_0_rgb"
            # ]
            # 列表长度：3，每个元素的形状均为 [batch_size, patch_len, paligemma_width]

            # 1.2 将当前摄像头的图像有效性掩码扩展到每一个 Patch Token
            # obs.image_masks[name]：[batch_size]，每个样本对应一个图像有效性标记
            # einops.repeat(..., "b -> b s", s=patch_len)：
            # 将每个样本的图像有效性标记复制 patch_len 次
            # image_mask：[batch_size, patch_len]
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )

            # 例如，处理完 "base_0_rgb" 后：
            # input_mask = [
            #     image_mask：[batch_size, patch_len]  # "base_0_rgb"
            # ]
            #
            # 处理完三路摄像头后：
            # input_mask = [
            #     image_mask：[batch_size, patch_len]  # "base_0_rgb"
            #     image_mask：[batch_size, patch_len]  # "left_wrist_0_rgb"
            #     image_mask：[batch_size, patch_len]  # "right_wrist_0_rgb"
            # ]
            #
            # 同一样本的所有 Patch Token 共享该图像的有效性标记：
            # 全部为 True：该样本的所有 Patch Token 均有效
            # 全部为 False：该样本的所有 Patch Token 均无效
            # 不同样本的掩码可以不同

            # 1.3 添加当前摄像头的注意力分组标记
            # 当前图像有 patch_len 个 Token，因此追加 patch_len 个 False
            # False：当前位置不启动新的注意力分组
            ar_mask += [False] * image_tokens.shape[1]

            # 例如，处理完 "base_0_rgb" 后：
            # ar_mask = [
            #     False, False, ..., False  # patch_len 个 False
            # ]
            # ar_mask 长度：patch_len
            #
            # 处理完 "left_wrist_0_rgb" 后：
            # ar_mask = [
            #     False, False, ..., False  # "base_0_rgb"，patch_len 个 False
            #     False, False, ..., False  # "left_wrist_0_rgb"，patch_len 个 False
            # ]
            # ar_mask 长度：2 * patch_len
            #
            # 处理完 "right_wrist_0_rgb" 后：
            # ar_mask = [
            #     False, False, ..., False  # "base_0_rgb"，patch_len 个 False
            #     False, False, ..., False  # "left_wrist_0_rgb"，patch_len 个 False
            #     False, False, ..., False  # "right_wrist_0_rgb"，patch_len 个 False
            # ]
            # ar_mask 长度：3 * patch_len

        # 此时，三路摄像头的图像特征、图像掩码和注意力分组标记均已添加完毕。
        #
        # tokens = [
        #     image_tokens：[batch_size, patch_len, paligemma_width]  # "base_0_rgb"
        #     image_tokens：[batch_size, patch_len, paligemma_width]  # "left_wrist_0_rgb"
        #     image_tokens：[batch_size, patch_len, paligemma_width]  # "right_wrist_0_rgb"
        # ]
        #
        # input_mask = [
        #     image_mask：[batch_size, patch_len]  # "base_0_rgb"
        #     image_mask：[batch_size, patch_len]  # "left_wrist_0_rgb"
        #     image_mask：[batch_size, patch_len]  # "right_wrist_0_rgb"
        # ]
        #
        # ar_mask = [
        #     False, False, ..., False  # "base_0_rgb"，patch_len 个 False
        #     False, False, ..., False  # "left_wrist_0_rgb"，patch_len 个 False
        #     False, False, ..., False  # "right_wrist_0_rgb"，patch_len 个 False
        # ]
        # ar_mask 长度：3 * patch_len

        # 2. 编码语言 Token；如果没有提供语言 Token，则跳过
        if obs.tokenized_prompt is not None:

            # 将语言 Token ID 转换为 PaliGemma 的词嵌入特征
            # obs.tokenized_prompt：[batch_size, seq_len]
            # tokenized_inputs：[batch_size, seq_len, paligemma_width]
            tokenized_inputs = self.PaliGemma.llm(
                # 指定 method="embed"，调用 llm 模块的 embed() 方法
                # 其余参数均作为 embed() 方法的参数传入
                obs.tokenized_prompt,
                method="embed",
            )

            # 2.1 将语言 Token 特征添加到 tokens 列表
            tokens.append(tokenized_inputs)

            # tokens = [
            #     image_tokens：[batch_size, patch_len, paligemma_width]  # "base_0_rgb"
            #     image_tokens：[batch_size, patch_len, paligemma_width]  # "left_wrist_0_rgb"
            #     image_tokens：[batch_size, patch_len, paligemma_width]  # "right_wrist_0_rgb"
            #     tokenized_inputs：[batch_size, seq_len, paligemma_width]  # 语言 Token
            # ]
            # 列表长度：4

            # 2.2 将语言 Token 的有效性掩码添加到 input_mask 列表
            # obs.tokenized_prompt_mask：[batch_size, seq_len]
            input_mask.append(obs.tokenized_prompt_mask)

            # input_mask = [
            #     image_mask：[batch_size, patch_len]  # "base_0_rgb"
            #     image_mask：[batch_size, patch_len]  # "left_wrist_0_rgb"
            #     image_mask：[batch_size, patch_len]  # "right_wrist_0_rgb"
            #     obs.tokenized_prompt_mask：[batch_size, seq_len]  # 语言 Token 掩码
            # ]
            #
            # 图像掩码的有效性由 obs.image_masks 决定
            # 语言掩码的有效性由 obs.tokenized_prompt_mask 决定
            # True：对应位置有效
            # False：对应位置无效

            # 2.3 添加语言 Token 的注意力分组标记
            # 语言序列长度为 seq_len，因此追加 seq_len 个 False
            # False：当前位置不启动新的注意力分组
            ar_mask += [False] * tokenized_inputs.shape[1]

            # 添加语言 Token 后：
            # ar_mask = [
            #     False, False, ..., False  # "base_0_rgb"，patch_len 个 False
            #     False, False, ..., False  # "left_wrist_0_rgb"，patch_len 个 False
            #     False, False, ..., False  # "right_wrist_0_rgb"，patch_len 个 False
            #     False, False, ..., False  # 语言 Token，seq_len 个 False
            # ]
            # ar_mask 长度：3 * patch_len + seq_len

        # 3. 沿序列维度拼接所有图像 Token 和语言 Token
        tokens = jnp.concatenate(tokens, axis=1)

        # 有语言 Token：
        # tokens：[batch_size, 3 * patch_len + seq_len, paligemma_width]
        #
        # 没有语言 Token：
        # tokens：[batch_size, 3 * patch_len, paligemma_width]
        #
        # prefix_len 表示最终前缀序列的总长度：
        # 有语言 Token：prefix_len = 3 * patch_len + seq_len
        # 没有语言 Token：prefix_len = 3 * patch_len

        # 4. 沿序列维度拼接所有图像掩码和语言掩码
        input_mask = jnp.concatenate(input_mask, axis=1)

        # 有语言 Token：
        # input_mask：[batch_size, 3 * patch_len + seq_len]
        #
        # 没有语言 Token：
        # input_mask：[batch_size, 3 * patch_len]
        #
        # True：对应的图像 Token 或语言 Token 有效
        # False：对应的图像 Token 或语言 Token 无效

        # 5. 将注意力分组标记列表转换为 JAX 数组
        ar_mask = jnp.array(ar_mask)

        # 有语言 Token：
        # ar_mask：[3 * patch_len + seq_len]
        #
        # 没有语言 Token：
        # ar_mask：[3 * patch_len]
        #
        # 三路图像 Token 的标记均为 False
        # 如果提供语言 Token，其标记也全部为 False
        # 后续由注意力掩码构造逻辑根据 ar_mask 生成实际的注意力掩码

        # 6. 返回前缀特征序列、有效性掩码和注意力分组标记
        return tokens, input_mask, ar_mask
    
    # embed_suffix()：将机器人状态、加噪动作和 Flow Matching 时间步编码为后缀特征序列，
    # 并生成对应的有效性掩码 input_mask、注意力分组标记 ar_mask 和 AdaRMSNorm 条件向量 adarms_cond。
    #
    # 输入：
    # obs：机器人观测数据，包含机器人状态等信息。
    # noisy_actions：[batch_size, action_horizon, action_dim]，加噪动作序列。
    # timestep：[batch_size]，Flow Matching 时间步。
    #
    # 返回：
    # tokens：[batch_size, suffix_len, action_expert_width]，后缀特征序列。
    # input_mask：[batch_size, suffix_len]，输入有效性掩码。
    # ar_mask：[suffix_len]，注意力分组标记。
    # adarms_cond：[batch_size, action_expert_width] 或 None，AdaRMSNorm 条件向量。

    # @at.typecheck：检查函数输入参数和返回值是否符合声明的类型及形状要求
    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,            # 机器人观测数据：图像、机器人状态、语言 Token 等
        noisy_actions: _model.Actions,      # 加噪动作序列：[batch_size, action_horizon, action_dim]
        timestep: at.Float[at.Array, "b"],  # Flow Matching 时间步：[batch_size]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],      # tokens：[batch_size, suffix_len, action_expert_width]
        at.Bool[at.Array, "b s"],           # input_mask：[batch_size, suffix_len]
        at.Bool[at.Array, " s"],            # ar_mask：[suffix_len]
        at.Float[at.Array, "b emb"] | None, # adarms_cond：[batch_size, action_expert_width] 或 None
    ]:
        tokens = []      # 保存状态 Token 和动作 Token 的特征
        input_mask = []  # 保存状态 Token 和动作 Token 的有效性掩码

        # 保存各个 Token 的自回归注意力分组标记，用于构造注意力掩码
        # 同组内可以相互注意；不同组之间，后组可以关注前组，前组不能关注后组
        # True：开启一个新的注意力分组
        # False：延续当前注意力分组
        ar_mask = []

        # 1. 普通 π0 添加状态 Token；π0.5 不在此处添加状态 Token
        if not self.pi05:

            # 将机器人状态映射到 Action Expert 的隐藏特征空间
            # obs.state：[batch_size, state_dim]，当前配置中 state_dim = action_dim
            # self.state_proj(obs.state)：[batch_size, action_expert_width]
            # [:, None, :]：在序列维度插入长度为 1 的维度
            # state_token：[batch_size, 1, action_expert_width]
            state_token = self.state_proj(obs.state)[:, None, :]

            # 将状态 Token 添加到 tokens 列表
            # tokens = [
            #     state_token：[batch_size, 1, action_expert_width]
            # ]
            tokens.append(state_token)

            # 创建状态 Token 的有效性掩码，True 表示该位置有效
            # state_mask：[batch_size, 1]，状态掩码（均为有效）
            input_mask.append(
                jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_)
            )

            # 添加状态 Token 的注意力分组标记
            # ar_mask = [True]，状态 Token 开启一个新的注意力分组
            ar_mask += [True]

        # 2. 将加噪动作从 action_dim 维映射到 Action Expert 的隐藏特征空间
        # noisy_actions：[batch_size, action_horizon, action_dim]
        # action_tokens：[batch_size, action_horizon, action_expert_width]
        action_tokens = self.action_in_proj(noisy_actions)

        # 3. 使用正弦-余弦位置编码，将 Flow Matching 时间步编码为向量
        # timestep：[batch_size]
        # time_emb：[batch_size, action_expert_width]
        time_emb = posemb_sincos(
            pos = timestep,
            embedding_dim = self.action_in_proj.out_features,  # 嵌入维度：action_expert_width
            min_period = 4e-3,                   # 最小周期
            max_period = 4.0,                    # 最大周期
        )

        # 4. 根据模型类型选择时间信息的注入方式

        # π0.5：不将时间嵌入直接拼接到动作 Token，而是通过 AdaRMSNorm 注入时间条件
        if self.pi05:

            # 4.1 通过时间 MLP 输入层处理时间嵌入
            # time_emb：[batch_size, action_expert_width]
            time_emb = self.time_mlp_in(time_emb)

            # 4.2 使用 Swish 激活函数，形状不变
            time_emb = nnx.swish(time_emb)

            # 4.3 通过时间 MLP 输出层进一步处理时间嵌入
            # time_emb：[batch_size, action_expert_width]
            time_emb = self.time_mlp_out(time_emb)

            # 4.4 使用 Swish 激活函数，形状不变
            time_emb = nnx.swish(time_emb)

            # 4.5 将加噪动作经过线性投影得到的特征，直接作为 Action Expert 的输入 Token
            # action_tokens：[batch_size, action_horizon, action_expert_width]
            action_expert_tokens = action_tokens

            # 4.6 将时间嵌入作为 AdaRMSNorm 的条件向量
            # adarms_cond：[batch_size, action_expert_width]
            adarms_cond = time_emb

        # 普通 π0：将时间嵌入与动作特征拼接，再经过 MLP 融合
        else:

            # 4.1 将时间嵌入沿动作序列维度重复 action_horizon 次
            # time_emb：[batch_size, action_expert_width]
            # time_tokens：[batch_size, action_horizon, action_expert_width]
            time_tokens = einops.repeat(
                time_emb,
                "b emb -> b s emb",
                s=self.action_horizon,
            )

            # 4.2 沿最后一个维度拼接动作特征与时间特征
            # action_tokens：[batch_size, action_horizon, action_expert_width]
            # time_tokens：[batch_size, action_horizon, action_expert_width]
            # action_time_tokens：[batch_size, action_horizon, 2 * action_expert_width]
            action_time_tokens = jnp.concatenate(
                [action_tokens, time_tokens],
                axis=-1,
            )

            # 4.3 通过 MLP 输入层，将拼接后的特征从 2 * action_expert_width 映射到 action_expert_width
            # [batch_size, action_horizon, 2 * action_expert_width]
            #     -> [batch_size, action_horizon, action_expert_width]
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)

            # 4.4 使用 Swish 激活函数，形状不变
            action_time_tokens = nnx.swish(action_time_tokens)

            # 4.5 通过 MLP 输出层进一步处理融合特征，维度保持为 action_expert_width
            # action_time_tokens：[batch_size, action_horizon, action_expert_width]
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)

            # 4.6 将融合动作信息与时间信息后的特征作为 Action Expert 的输入 Token
            # action_expert_tokens：[batch_size, action_horizon, action_expert_width]
            action_expert_tokens = action_time_tokens

            # 4.7 普通 π0 不使用 AdaRMSNorm 条件向量
            adarms_cond = None

        # 5. 将动作 Token 添加到 tokens 列表
        # 普通 π0：
        # tokens = [
        #     state_token：[batch_size, 1, action_expert_width]
        #     action_expert_tokens：[batch_size, action_horizon, action_expert_width]
        # ]
        #
        # π0.5：
        # tokens = [
        #     action_expert_tokens：[batch_size, action_horizon, action_expert_width]
        # ]
        tokens.append(action_expert_tokens)

        # 6. 创建动作 Token 的有效性掩码
        # action_mask：[batch_size, action_horizon]，动作掩码（均为有效）
        input_mask.append(
            jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_)
        )

        # 7. 添加动作 Token 的注意力分组标记
        # 第一个动作 Token 标记为 True，开启新的注意力分组
        # 后续动作 Token 标记为 False，延续当前注意力分组
        #
        # 普通 π0（pi05=False）：
        # ar_mask = [
        #     True，  # state_token：状态 Token 组
        #     True，  # 动作组的第 1 个 Token
        #     False， # 动作组的第 2 个 Token
        #     ...,
        #     False， # 动作组的第 action_horizon 个 Token
        # ]
        #
        # π0.5（pi05=True）：
        # ar_mask = [
        #     True，  # 动作组的第 1 个 Token
        #     False， # 动作组的第 2 个 Token
        #     ...,
        #     False， # 动作组的第 action_horizon 个 Token
        # ]
        ar_mask += [True] + ([False] * (self.action_horizon - 1))

        # 8. 沿序列维度拼接状态 Token 和动作 Token
        tokens = jnp.concatenate(tokens, axis=1)
        # 普通 π0：[batch_size, 1 + action_horizon, action_expert_width]
        # π0.5：[batch_size, action_horizon, action_expert_width]

        # 9. 沿序列维度拼接状态掩码和动作掩码
        input_mask = jnp.concatenate(input_mask, axis=1)
        # 普通 π0：[batch_size, 1 + action_horizon]
        # π0.5：[batch_size, action_horizon]

        # 10. 将注意力分组标记列表转换为 JAX 数组
        ar_mask = jnp.array(ar_mask)
        # 普通 π0：[1 + action_horizon]
        # π0.5：[action_horizon]

        # 11. 返回后缀特征序列、有效性掩码、注意力分组标记和 AdaRMSNorm 条件向量
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, 
        rng: at.KeyArrayLike,  # 随机数秘钥
        observation: _model.Observation, # 机器人观测数据
        actions: _model.Actions, # 真实动作序列: [batch_size, action_horizon, action_dim]
        *, 
        train: bool = False # 是否处于训练模式
    ) -> at.Float[at.Array, "*b ah"]: # 反悔损失: []

        # 将随机数秘钥拆分成三个独立的 key, 分别用于不同的随机操作
        # 1. preprocess_rng: 观测数据预处理所用的随机数秘钥
        # 2. noise_rng: 生成动作噪声所用的随机数秘钥
        # 3. time_rng: 采样 Flow Matching 时间步所用的随机数秘钥
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)

        # 对机器人观测数据进行预处理
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        # 获取动作序列的批次维度
        # batch_shape = [batch_size]
        batch_shape = actions.shape[:-2]

        # 为真实动作生成标准正态分布噪声
        # noise: [batch_size, action_horizon, action_dim]
        noise = jax.random.normal(noise_rng, actions.shape)

        # 从 Beta 分布中采样 Flow Matching 时间步
        # 重点：time 不是机器人执行动作的时间，而是 Flow Matching 在噪声与真实动作之间插值的时间
        # 1. jax.random.beta(...): 从参数 alpha=1.5, beta=1 的 Beta 分布中采样
        #    1. shape: [batch_size]
        #    2. 值域: [0, 1]
        # 2. * 0.999 + 0.001: 将值域缩放到 [0.001, 1]，从而避免时间步过于接近0，也避免采样到真实的 0
        # 3. time: [batch_size], 值域 [0.001, 1]
        time = jax.random.beta(
            key = time_rng, 
            a = 1.5, 
            b = 1, 
            shape = batch_shape
        ) * 0.999 + 0.001

        # time: [batch_size, 1, 1] 新增两个维度对应 action_horizon 和 action_dim
        time_expanded = time[..., None, None]

        # 构造 Flow Matching 中间状态 x_t
        # 1. time_expanded: [batch_size, 1, 1]
        # 2. noise: [batch_size, action_horizon, action_dim]
        # 3. actions: [batch_size, action_horizon, action_dim]
        # 4. x_t = t * noise + (1 - t) * actions
        #    1. t->0 时，x_t -> actions
        #    2. t->1 时，x_t -> noise
        # x_t: [batch_size, action_horizon, action_dim]
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        # 构造 Flow Mathcing 的目标速度 u_t
        # 根据线性插值路径，目标速度为 noise - actions
        # u_t: [batch_size, action_horizon, action_dim]，用于监督模型预测的速度
        u_t = noise - actions

        # 前缀编码：将摄像头图像 + 语言 Token 编码为 PaliGemma 中 Gemma 的输入特征
        # 带语言token: prefix_len = 3 * patch_len + seq_len
        # 不带语言token: prefix_len = 3 * patch_len
        # prefix_tokens：[batch_size, prefix_len, paligemma_width]
        # prefix_mask：[batch_size, prefix_len]
        # prefix_ar_mask：[prefix_len]
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # 后缀编码：将机器人状态（普通 π0）+ 加噪动作 x_t + 时间信息编码为 Action Expert 中 Gemma 的输入特征
        # 普通 π0：将时间嵌入与动作 Token 拼接，再通过 MLP 融合，直接将时间信息注入动作特征
        # π0.5：不将时间嵌入拼接到动作 Token，而是将时间嵌入经过时间 MLP 处理后，编码为 AdaRMSNorm 的条件向量
        # 
        # 带状态: suffix_len = 1 + action_horizon
        # 不带状态: suffix_len = action_horizon
        # suffix_tokens：[batch_size, suffix_len, action_expert_width]
        # suffix_mask：[batch_size, suffix_len]
        # suffix_ar_mask：[suffix_len]
        # adarms_cond（AdaRMSNorm 的条件向量）：
        #   π0.5：[batch_size, action_expert_width]，用于将时间条件注入 Action Expert 内部的 AdaRMSNorm
        #   普通 π0：None，时间信息已融合到动作 Token 中
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

        # 拼接 前缀编码 与 后缀编码 的有效性掩码
        # prefix_mask：[batch_size, prefix_len]
        # suffix_mask：[batch_size, suffix_len]
        # input_mask: [batch_size, prefix_len + suffix_len]
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)

        # 拼接 前缀编码 与 后缀编码 的注意力分组标记
        # prefix_ar_mask：[prefix_len]
        # suffix_ar_mask：[suffix_len]
        # ar_mask = [prefix_len + suffix_len]
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)

        # 根据前后缀编码的 有效性掩码 和 注意力分组标记 构造实际的注意力掩码
        # input_mask: [batch_size, prefix_len + suffix_len]
        # ar_mask = [prefix_len + suffix_len]
        # attn_mask: [batch_size, prefix_len + suffix_len, prefix_len + suffix_len]
        attn_mask = make_attn_mask(input_mask, ar_mask)

        # 给 token 进行位置编号
        # 1. 保证第一个有效 Token 的位置索引一定为 0，无论前面是否存在 Padding Token
        # 2. 保证后续有效 Token 的位置索引连续递增，Padding Token | 无效 token 不占用位置编号
        # input_mask: [batch_size, prefix_len + suffix_len]
        # jnp.cumsum(...): 布尔值会按照数值参与累积求和，True = 1，False = 0
        positions = jnp.cumsum(input_mask, axis=1) - 1

        # 训练阶段：通过 Prefix + Suffix 的联合 Attention 提取特征
        # Attention([Q_prefix, Q_suffix], [K_prefix, K_suffix], [V_prefix, V_suffix])
        # prefix_out: [batch_size, prefix_len, paligemma_width]
        # suffix_out: [batch_size, suffix_len, action_expert_width]
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )

        # suffix_out[:, -self.action_horizon :]：[batch_size, action_horizon, action_expert_width]
        # 1. suffix_out：[batch_size, suffix_len, action_expert_width]
        # 2. -self.action_horizon：选取最后 action_horizon 个 Token
        #    1) π0：Suffix 包含 1 个 State Token 和 action_horizon 个 Action Token，
        #       suffix_len = 1 + action_horizon，因此需要排除第一个 State Token
        #    2) π0.5：State 作为离散 Token 输入 Prefix，Suffix 仅包含 Action Token，
        #       suffix_len = action_horizon，因此直接选取全部 Action Token
        # self.action_out_proj(...): [batch_size, suffix_len, action_dim]
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # Flow Matching 均方误差
        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        # rng: 随机数秘钥，用于生成初始高斯噪声
        rng: at.KeyArrayLike,
        # observation: 机器人观测数据
        observation: _model.Observation,
        *,
        # Flow Matching 推理的积分步数，默认为 10 步
        num_steps: int | at.Int[at.Array, ""] = 10,
        # 初始的高斯噪声，可由外部传入
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:

        # 预处理机器人观测数据
        observation = _model.preprocess_observation(None, observation, train=False)

        # 计算积分步长
        # t == 1: A_t = noise
        # t == 0: A_t = real Action
        # dt = -1.0 / 10 = -0.1 即每次走 -0.1 步长
        dt = -1.0 / num_steps

        # 获取 batch_size
        batch_size = observation.state.shape[0]

        # 如果没有传入噪声，则随机生成标准高斯噪声
        if noise is None:

            # noise: [batch_size, action_horizon, action_dim]
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # 前缀编码：将摄像头图像 + 语言 Token 编码为 PaliGemma 中 Gemma 的输入特征
        # 带语言token: prefix_len = 3 * patch_len + seq_len
        # 不带语言token: prefix_len = 3 * patch_len
        # prefix_tokens：[batch_size, prefix_len, paligemma_width]
        # prefix_mask：[batch_size, prefix_len]
        # prefix_ar_mask：[prefix_len]
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # 注意力机制掩码 prefix_attn_mask: [batch_size, prefix_len]
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        
        # 给 token 进行位置编号
        # 1. 保证第一个有效 Token 的位置索引一定为 0，无论前面是否存在 Padding Token
        # 2. 保证后续有效 Token 的位置索引连续递增，Padding Token | 无效 token 不占用位置编号
        # prefix_mask：[batch_size, prefix_len]
        # jnp.cumsum(...): 布尔值会按照数值参与累积求和，True = 1，False = 0
        positions = jnp.cumsum(prefix_mask, axis=1) - 1

        # 只计算 Prefix 的 Self-Attention
        # kv_cache = (key_cache, value_cache)
        #
        # key_cache, value_cache：[depth, batch_size, prefix_len, num_kv_heads, head_dim]
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        # 根据当前带噪动作 x_t，预测动作变化的速度 v_t，然后沿着速度方向更新动作，使其逐渐从高斯噪声变成机器人可以执行的动作
        # carry = (x_t, time)
        # x_t: [batch_size, action_horizon, action_dim] 当前带噪声的动作
        # time: 当前的积分时间 t，初始值为 1
        def step(carry):
            # x_t: [batch_size, action_horizon, action_dim] 当前带噪声的动作
            # time: 当前的积分时间 t，初始值为 1
            x_t, time = carry

            # 后缀编码：将机器人状态（普通 π0）+ 加噪动作 x_t + 时间信息编码为 Action Expert 中 Gemma 的输入特征
            # 普通 π0：将时间嵌入与动作 Token 拼接，再通过 MLP 融合，直接将时间信息注入动作特征
            # π0.5：不将时间嵌入拼接到动作 Token，而是将时间嵌入经过时间 MLP 处理后，编码为 AdaRMSNorm 的条件向量
            # 
            # 带状态: suffix_len = 1 + action_horizon
            # 不带状态: suffix_len = action_horizon
            # suffix_tokens：[batch_size, suffix_len, action_expert_width]
            # suffix_mask：[batch_size, suffix_len]
            # suffix_ar_mask：[suffix_len]
            # adarms_cond（AdaRMSNorm 的条件向量）：
            #   π0.5：[batch_size, action_expert_width]，用于将时间条件注入 Action Expert 内部的 AdaRMSNorm
            #   普通 π0：None，时间信息已融合到动作 Token 中
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size) # [batch_size]
            )

            # 后缀编码的注意力掩码 suffix_attn_mask: [batch_size, suffix_len, suffix_len]
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)

            # 构建 Suffix → Prefix 的注意力掩码（这里可不是 prefix token 的自注意力掩码）：
            # 1. 注意力可见性：所有 Suffix Token 都允许关注所有 Prefix Token，无需额外限制。
            # 2. Token 有效性：Prefix 中可能存在无效 Token，需要通过 prefix_mask 排除。
            # 因此，只需将 prefix_mask 复制 suffix_len 次，作为 Suffix → Prefix 的注意力掩码。
            #
            # suffix_tokens：[batch_size, suffix_len, action_expert_width]
            # suffix_tokens.shape[1] = suffix_len
            # prefix_mask：[batch_size, prefix_len]
            # prefix_attn_mask：[batch_size, suffix_len, prefix_len]
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])

            # 构建 Suffix → Prefix + Suffix 的注意力掩码
            #
            # prefix_attn_mask：[batch_size, suffix_len, prefix_len]
            # suffix_attn_mask：[batch_size, suffix_len, suffix_len]
            #
            # axis=-1：沿最后一个维度（Key 维度）进行拼接
            #
            # full_attn_mask：[batch_size, suffix_len, prefix_len + suffix_len]
            # Query：Suffix Token
            # Key：Prefix Token + Suffix Token
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            # 断言: full_attn_mask.shape == [batch_size, suffix_len, prefix_len + suffix_len]
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )

            # 为 Suffix Token 生成位置索引，使其接在有效 Prefix Token 后面，用于 RoPE
            #
            # 1. 计算每个样本有效 Prefix Token 的数量：
            #    prefix_mask：[batch_size, prefix_len]
            #    jnp.sum(prefix_mask, axis=-1)：[batch_size]
            #    jnp.sum(...)[:, None]：[batch_size, 1]，方便广播
            #
            # 2. 为 Suffix Token 生成相对位置索引：
            #    suffix_mask：[batch_size, suffix_len]
            #    jnp.cumsum(suffix_mask, axis=-1) - 1：[batch_size, suffix_len]
            #    True = 1，False = 0
            #    (1) 保证第一个有效 Suffix Token 的相对位置索引为 0，即使前面存在 Padding Token
            #    (2) 保证后续有效 Suffix Token 的相对位置索引连续递增，Padding Token 不增加位置计数
            #
            # 3. 生成 Suffix Token 在 Prefix + Suffix 序列中的最终位置索引：
            #    将有效 Prefix Token 的数量加到 Suffix 的相对位置索引上
            #    使第一个有效 Suffix Token 的位置紧接在最后一个有效 Prefix Token 后面
            #
            # positions：[batch_size, suffix_len]
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            # 推理阶段：计算 Suffix Token 的 Attention
            # Attention(Q_suffix, [K_prefix, K_suffix], [V_prefix, V_suffix])
            #
            # Q_suffix、K_suffix、V_suffix：由当前 Suffix Token 生成
            # K_prefix、V_prefix：从 kv_cache 中读取
            #
            # prefix_out：None，因为 Prefix 的 K、V 已缓存，不再计算 Prefix 输出
            # suffix_out：[batch_size, suffix_len, action_expert_width]
            #   π0：包含 1 个 State Token 和 action_horizon 个 Action Token 的输出特征
            #   π0.5：仅包含 action_horizon 个 Action Token 的输出特征
            #
            # 注意：suffix_out 是 Transformer 输出的隐藏特征，并非预测速度场 v_t
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )

            # 断言 Prefix 输出为 None，因为当前只计算 Suffix
            assert prefix_out is None

            # 从 suffix_out 中提取 Action Token 的隐藏特征，并预测速度场 v_t
            #
            # suffix_out：[batch_size, suffix_len, action_expert_width]
            #
            # suffix_out[:, -self.action_horizon:]：
            #   选取最后 action_horizon 个 Token
            #
            #   1. π0：
            #      suffix_len = 1 + action_horizon
            #      Suffix 包含 1 个 State Token 和 action_horizon 个 Action Token
            #      因此需要排除第一个 State Token，只保留 Action Token
            #
            #   2. π0.5：
            #      suffix_len = action_horizon
            #      State 已作为离散 Token 输入 Prefix
            #      因此直接选取全部 Action Token
            #
            # suffix_out[:, -self.action_horizon:]：
            #   [batch_size, action_horizon, action_expert_width]
            #
            # self.action_out_proj(...)：
            #   将 Action Token 的隐藏特征从 action_expert_width 映射到 action_dim
            #
            # v_t：[batch_size, action_horizon, action_dim]
            #   表示整个动作序列在当前 Flow Matching 时间 t 下的预测速度场
            #   不是最终动作 A，而是用于后续更新带噪动作 x_t
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])

            # 使用 Euler（欧拉）积分法，根据预测速度场 v_t 更新动作和时间
            #
            # num_steps = 10，因此 dt = -1.0 / 10 = -0.1
            #
            # 1. x_t + dt * v_t：
            #    x_t：[batch_size, action_horizon, action_dim]，当前带噪动作序列
            #    v_t：[batch_size, action_horizon, action_dim]，当前预测的速度场
            #    x_next = x_t - 0.1 * v_t
            #    每次更新整个动作序列，经过 10 次积分，逐步从高斯噪声生成最终动作
            #
            # 2. time + dt：
            #    time：当前 Flow Matching 时间，初始值为 1.0
            #    time_next = time - 0.1
            #    时间依次为 1.0 -> 0.9 -> 0.8 -> ... -> 0.0
            #
            # 每次积分都需要根据更新后的 x_t 和 time 重新预测 v_t
            #
            # 返回：(x_next, time_next)
            #   x_next：[batch_size, action_horizon, action_dim]
            #   time_next：标量
            return x_t + dt * v_t, time + dt

        # 判断是否还需要继续进行欧拉积分
        def cond(carry):
            # x_t: [batch_size, action_horizon, action_dim] 当前带噪声的动作
            # time: 当前的积分时间 t
            # 随着逐渐积分，t 逐渐减小 1.0 -> 0.9 -> 0.8 -> ... -> 0.2 -> 0.1 -> 0.0(完成)
            x_t, time = carry

            # dt = -1.0 / 10 = -0.1
            # -dt / 2 = 0.05 
            # 也就是说，当 time >= 0.05 时，还没有积分完全，所以需要继续积分
            return time >= -dt / 2

        # 执行 JAX 的 While 循环
        # 执行过程（num_steps = 10，dt = -0.1）：
        #   1. cond() 判断 time >= 0.05 是否成立
        #   2. 成立则执行 step()，更新 x_t = x_t - 0.1 * v_t，time = time - 0.1
        #   3. 重复执行，直到 time 接近 0，停止循环
        #
        # 返回值：(x_0, time)
        #   x_0：[batch_size, action_horizon, action_dim]，经过 10 次积分生成的最终动作序列
        #   time：标量，最终时间约为 0.0，使用 _ 忽略
        x_0, _ = jax.lax.while_loop(
            # 1. 循环条件函数
            cond_fun = cond, 
            # 2. 循环体函数
            body_fun = step, 
            # 3. 循环初始状态
            init_val = (noise, 1.0)
        )

        # 返回最终预测的动作序列 x_0
        # Flow Matching 时间 t：从 1.0 逐步减小到 0.0
        # 经过 10 次 Euler 积分，将初始高斯噪声逐步转换为预测动作
        # x_0：[batch_size, action_horizon, action_dim]
        return x_0
