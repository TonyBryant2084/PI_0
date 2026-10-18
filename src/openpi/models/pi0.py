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


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:·
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)

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
            timestep,
            self.action_in_proj.out_features,  # 嵌入维度：action_expert_width
            min_period=4e-3,                   # 最小周期
            max_period=4.0,                    # 最大周期
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
        # prefix_tokens：[batch_size, prefix_len, paligemma_width]
        # prefix_mask：[batch_size, prefix_len]
        # prefix_ar_mask：[prefix_len]
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # 后缀编码：将机器人状态（普通 π0）+ 加噪动作 x_t + 时间信息编码为 Action Expert 中 Gemma 的输入特征
        # 普通 π0：将时间嵌入与动作 Token 拼接，再通过 MLP 融合，直接将时间信息注入动作特征
        # π0.5：不将时间嵌入拼接到动作 Token，而是将时间嵌入经过时间 MLP 处理后，编码为 AdaRMSNorm 的条件向量
        #
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
        attn_mask = make_attn_mask(input_mask, ar_mask)


        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
