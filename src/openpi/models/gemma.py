# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

# Paligemma 的词表大小：257152
PALIGEMMA_VOCAB_SIZE = 257_152

# @dataclasses.dataclass：
#   1) Python 标准库提供的通用数据类装饰器，适用于普通 Python 程序
#
#   2) 自动生成 __init__() 等常用方法，简化类的定义
#
#   3) 类中声明的字段会被识别为 dataclass 字段，例如：
#      dtype: str = "bfloat16"
#      其中 dtype 是字段名，str 是类型标注，"bfloat16" 是默认值
#
#   4) 创建实例时，自动生成的 __init__() 会将传入的参数赋给实例属性
#      如果没有传入对应参数，则使用字段声明中的默认值（前提是该字段有默认值）
#
#   5) 对于声明了默认值的字段，通常可以通过类名访问其默认值
#      对于未声明默认值的字段，通常不能直接通过类名访问其值
#      创建实例并赋值后，可以通过实例访问该属性
@dataclasses.dataclass
# 模型配置参数
class Config:

    # Transformer 隐藏层的维度，即每个 Token 对应的特征向量维度
    width: int

    # Transformer 的层数，即模型中堆叠的 Transformer Block 数量
    depth: int

    # 前馈神经网络（MLP）的中间隐藏层维度
    mlp_dim: int

    # Query（查询向量）的注意力头数量
    num_heads: int

    # Key（键向量）和 Value（值向量）的注意力头数量
    num_kv_heads: int

    # 每个注意力头的特征维度
    head_dim: int

    # LoRA 的配置：
    # 1. 对于不可变对象（str、int、float、tuple）：可以直接写 = "默认值"，因为它们无法被修改，只能被替换，不会互相影响。
    # 2. 对于可变对象（list、dict、set）：不能直接写 = [] 或 = {}，否则所有实例会共享同一个对象，改一个就全改了。
    # 3. 使用 dataclasses.field(default_factory=dict)：每次新建实例时自动调用 dict() 生成一个全新的空字典，确保每个实例都有独立的字典，互不干扰
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)

# 模型变体名
# Literal[...]：限定该类型只能取这几个固定字符串之一
Variant = Literal[
    "dummy",           # 测试用超小模型
    "gemma_300m",      # 约 3.11 亿参数的模型
    "gemma_300m_lora", # 约 3.11 亿参数的模型，支持 LoRA
    "gemma_2b",        # 约 20 亿参数的模型
    "gemma_2b_lora",   # 约 20 亿参数的模型，支持 LoRA
]


# 根据模型变体名称获取对应的配置参数
def get_config(variant: Variant) -> Config:

    # 测试用超小模型
    if variant == "dummy":
        return Config(
            width=64,       # 隐藏层维度：每个 Token 的特征向量维度
            depth=4,        # Transformer 层数：堆叠的 Transformer Block 数量
            mlp_dim=128,    # MLP 中间层维度
            num_heads=8,    # Query（查询向量）的注意力头数量
            num_kv_heads=1, # Key（键向量）和 Value（值向量）的注意力头数量
            head_dim=16,    # 每个注意力头的特征维度
        )

    # 约 3.11 亿参数的模型
    if variant == "gemma_300m":
        return Config(
            width=1024,      # 隐藏层维度
            depth=18,        # Transformer 层数
            mlp_dim=4096,    # MLP 中间层维度
            num_heads=8,     # Query（查询向量）的注意力头数量
            num_kv_heads=1,  # Key（键向量）和 Value（值向量）的注意力头数量
            head_dim=256,    # 每个注意力头的特征维度
        )  # 模型参数量约为 311M，即 3.11 亿

    # 约 3.11 亿参数的模型，支持 LoRA
    if variant == "gemma_300m_lora":
        return Config(
            width=1024,      # 隐藏层维度
            depth=18,        # Transformer 层数
            mlp_dim=4096,    # MLP 中间层维度
            num_heads=8,     # Query（查询向量）的注意力头数量
            num_kv_heads=1,  # Key（键向量）和 Value（值向量）的注意力头数量
            head_dim=256,    # 每个注意力头的特征维度
            # LoRA 配置
            lora_configs={
                # Attention 的 LoRA 配置：秩为 32，缩放参数为 32.0
                "attn": lora.LoRAConfig(rank=32, alpha=32.0),  
                # FeedForward 的 LoRA 配置：秩为 32，缩放参数为 32.0
                "ffn": lora.LoRAConfig(rank=32, alpha=32.0),   
            },
        ) 

    # 约 20 亿参数的模型
    if variant == "gemma_2b":
        return Config(
            width=2048,      # 隐藏层维度
            depth=18,        # Transformer 层数
            mlp_dim=16_384,  # MLP 中间层维度
            num_heads=8,     # Query（查询向量）的注意力头数量
            num_kv_heads=1,  # Key（键向量）和 Value（值向量）的注意力头数量
            head_dim=256,    # 每个注意力头的特征维度
        )

    # 约 20 亿参数的模型，支持 LoRA
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,      # 隐藏层维度
            depth=18,        # Transformer 层数
            mlp_dim=16_384,  # MLP 中间层维度
            num_heads=8,     # Query（查询向量）的注意力头数量
            num_kv_heads=1,  # Key（键向量）和 Value（值向量）的注意力头数量
            head_dim=256,    # 每个注意力头的特征维度
            # LoRA 配置
            lora_configs={
                # Attention 的 LoRA 配置：秩为 16，缩放参数为 16.0
                "attn": lora.LoRAConfig(rank=16, alpha=16.0),  
                # FeedForward 的 LoRA 配置：秩为 16，缩放参数为 16.0
                "ffn": lora.LoRAConfig(rank=16, alpha=16.0),   
            },
        )

    # 模型变体名称不受支持时抛出异常
    raise ValueError(f"Unknown variant: {variant}")  

# @at.typecheck 是一个类型检查装饰器，用于检查函数调用是否符合声明的类型要求。
@at.typecheck
# RMSNorm（Root Mean Square Layer Normalization，方均根归一化）
# 是 Transformer 中的一种归一化层。
# 它只根据特征的均方根进行缩放，不像 LayerNorm 那样减去均值。
class RMSNorm(nn.Module):

    # @nn.compact 允许在 __call__() 方法内部创建参数或子模块。
    # 例如 self.param() 和 nn.Dense() 都可以在该方法中使用。
    @nn.compact
    # 执行 均方根归一化操作
    def __call__(
        self, 
        # 输入 x：[batch_size, seq_len, width]
        x, 
        # 条件信息 cond：[batch_size, cond_dim]，用于生成 Adaptive RMSNorm 的 scale、shift 和 gate。
        #
        # 1. 如果 cond=None，则使用 Regular RMSNorm：
        #    Regular RMSNorm(x) = x / sqrt(mean(x^2) + ε) ⊙ (1 + scale)
        #    其中 scale 是 RMSNorm 自身的可学习参数，形状为 [width]。
        #
        # 2. 如果 cond != None，则使用 Adaptive RMSNorm：
        #    Adaptive RMSNorm(x, cond) = x / sqrt(mean(x^2) + ε) ⊙ (1 + scale) + shift
        #    其中 scale、shift 和 gate 由 cond 经过 Dense 层生成。
        #
        # 3. gate 不参与 RMSNorm 的归一化计算，而是在后续 Attention 残差连接中使用：
        #    x_new = x + gate ⊙ Attention(Adaptive RMSNorm(x, cond))
        #    gate 用于控制 Attention 输出对原始输入 x 的残差更新贡献。
        #
        # 4. 运算符说明：
        #    1. *：逐元素相乘（Element-wise Multiplication），支持自动广播，参与运算的对象可以是标量、向量、矩阵或更高维张量，数学公式中通常用 ⊙ 表示逐元素相乘。
        #    2. @：矩阵乘法（Matrix Multiplication），遵循矩阵乘法的维度匹配规则。
        cond
    ):
        # 保存原始输入类型
        dtype = x.dtype  

        # 计算每个 token 隐藏特征表示 的 平方均值
        # 1. x.astype(jnp.float32)：先将输入转为 float32，提高平方和均值计算的数值精度。        
        # 2. axis = -1：沿着最后一维（隐藏特征表示）进行求均值
        # 3. keepdims = True：保留最后一个维度为 1
        # var：[batch_size, seq_len, 1]
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  

        # 计算 x / RMS = x / √(mean(x^2) + epsilon)
        # epsilon = 1e-06：避免分母为 0
        # normed_inputs：[batch_size, seq_len, width]
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))

        # Regular RMSNorm：
        if cond is None:
            # 创建可学习参数 scale，shape：[width]
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))

            # Regular RMSNorm(x) = x / sqrt(mean(x^2) + ε) ⊙ (1 + scale)
            #   1. normed_inputs = x / sqrt(mean(x^2) + ε)：[batch_size, seq_len, width]
            #   2. (1 + scale)：[width]
            #   3. 广播：从右向左对齐维度（维度数量不足时在左侧补 1），再将维度为 1 的维度扩展至匹配的大小
            #           1. normed_inputs：[batch_size, seq_len, width]
            #           2. (1 + scale)：[width] -> [1, 1, width] -> [batch_size, seq_len, width]
            #   4. 结果：[batch_size, seq_len, width]
            normed_inputs = normed_inputs * (1 + scale)

            # 以原始输入类型进行返回，因为是 Regular RMSNorm 所以没有 gate
            # normed_inputs：[batch_size, seq_len, width]
            return normed_inputs.astype(dtype), None 
        
        # Adaptive RMSNorm：
        # 输入的条件信息 cond：[batch_size, cond_dim]
        # cond -> Dense(全连接层) -> modulation
        # modulation 是 scale, shift, gate 三组调制参数的集合：[batch_size, 3 * width]
        modulation = nn.Dense(
            # 输出特征维度：3 * width，用于生成 scale、shift、gate 三组参数。
            features=x.shape[-1] * 3,

            # 将全连接层的权重初始化为全 0。
            kernel_init=nn.initializers.zeros,

            # 指定全连接层的计算数据类型；输入和权重按该类型参与计算，
            # 输出通常也是该类型。参数的存储类型由 param_dtype 控制。
            dtype=dtype,
        )(cond) 

        # 将 modulation 拆分为 scale, shift, gate
        # modulation[:, None, :]：[batch_size, 1, 3 * width]
        # scale, shift, gate：[batch_size, 1, width]
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)

        # normed_inputs：[batch_size, seq_len, width]
        # (1 + scale)：[batch_size, 1, width] -> [batch_size, seq_len, width]
        # shift：[batch_size, 1, width] -> [batch_size, seq_len, width]
        # 结果：normed_inputs：[batch_size, seq_len, width]
        normed_inputs = normed_inputs * (1 + scale) + shift  

        # 以原始输入类型进行返回
        # normed_inputs.astype(dtype)：[batch_size, seq_len, width]
        # gate：[batch_size, 1, width]
        return normed_inputs.astype(dtype), gate

# @at.typecheck 是一个类型检查装饰器，用于检查函数调用是否符合声明的类型要求。
@at.typecheck
# 词嵌入层
class Embedder(nn.Module):

    # 词表大小：词表中的 token 的总数量
    vocab_size: int

    # 词嵌入维度：每个 token 对应的特征向量的维度
    embed_dim: int

    # 定义子模块
    def setup(self):

        # 创建可学习的词嵌入表
        # input_embedding_table：[vocab_size, embed_dim]
        # 每个 token 对应 一个 embedding 向量
        self.input_embedding_table = self.param(
            # 1. 参数名
            "input_embedding",
            # 2. 初始化方式：使用正态分布初始化
            nn.initializers.normal(),
            # 3. 参数形状：[vocab_size, embed_dim]
            (self.vocab_size, self.embed_dim),
        )

    # 编码 token -> embedding 向量
    def encode(self, x):

        # 使用 token ID 作为词嵌入表的行索引，取出对应的 embedding
        # input_embedding_table：[vocab_size, embed_dim]
        #
        # x：[batch_size, seq_len]
        # 输出：[batch_size, seq_len, embed_dim]
        x = self.input_embedding_table[(x,)]

        # 将 embedding 向量乘以 sqrt(embed_dim)，调整其数值尺度。
        # astype(dtype) 将缩放因子转换为输入 x 的数据类型。   
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)

        return x

    # 将隐藏特征向量映射为词表中每个 Token 对应的 logits，而不是直接得到 Token ID
    def decode(self, x):

        # x：[embed_dim]，在矩阵乘法中就当成 1 x embed_dim，而不是广播
        # input_embedding_table.T：[embed_dim, vocab_size]
        # x @ input_embedding_table.T：1 x vocab_size = [vocab_size]
        return jnp.dot(x, self.input_embedding_table.T)

# 给每个 expert 的模块 起名字
def _name(
    # 1. 模块名称
    name, 
    # 2. expert 的索引
    i
):
    # 如果 i==0 则直接返回原始名称
    if i == 0:
        return name

    # 如果 i!=0，则在原始名称基础上添加下标
    return f"{name}_{i}"

# RoPE (Rotary Positional Embedding，旋转位置编码) 
def _apply_rope(
    # 1. 需要进行旋转的位置向量：[batch_size, seq_len, num_heads or num_kv_heads, head_dim]
    x, 
    # 2. *：前侧参数用关键字或者位置传参，后者只能通过关键字传参
    *, 
    # 3. 每个 token 的位置：[batch_size, seq_len]
    positions, 
    # 4. 最大波长
    max_wavelength=10_000
):
    # 频率指数：
    # freq_exponents：[head_dim / 2]
    # 公式：freq_exponents[i] = 2i / head_dim, i = 0 ~ head_dim/2 - 1
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)

    # 时间尺度：处于相同位置的 token，时间尺度越大，旋转角度越小
    # timescale：[head_dim / 2]
    # 公式：timescale[i] = max_wavelength ** (2i / head_dim)
    #
    # 每个频率维度对应的 timescale 是固定的，与 token 的位置和具体内容无关，每个 token 有 head_dim/2 个频率维度
    timescale = max_wavelength**freq_exponents

    # 旋转角度：
    # positions[..., None]：[batch_size, seq_len, 1]
    # timescale[None, None, :]：[1, 1, head_dim / 2]
    # positions[..., None] / timescale[None, None, :]：[batch_size, seq_len, head_dim / 2]
    # radians：[batch_size, seq_len, head_dim / 2]
    # b, l, i：第 b 个样本的第 l 个 token 的第 i 个频率维度
    # 公式：radians[b, l, i] = position[b, l] / timescale[i]
    #
    # 每个 样本 的每个 token 都有一个固定的位置 positions[b, l]
    # 每个 token 的每个频率维度对应的 timesacle[i] 是固定的，与 token 的位置和具体内容无关
    # 通过 positions[b, l] 和 timescale[i] 即可算出每个 样本 的每个 token 的每个 频率维度 对应的 旋转角度
    radians = positions[..., None] / timescale[None, None, :]

    # radians：[batch_size, seq_len, 1, head_dim / 2]
    radians = radians[..., None, :]

    assert radians.dtype == jnp.float32

    # 位置编码 与 头 无关
    # sin, cos：[batch_size, seq_len, 1, head_dim / 2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)

    # x1, x2：[batch_size, seq_len, num_heads_or_kv_heads, head_dim / 2]
    x1, x2 = jnp.split(x, 2, axis=-1)

    # res：[batch_size, seq_len, num_heads_or_kv_heads, head_dim]   
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)

    assert res.dtype == jnp.float32

    # res：[batch_size, seq_len, num_heads_or_kv_heads, head_dim]   
    return res.astype(x.dtype)


# @at.typecheck 是一个类型检查装饰器，用于检查函数调用是否符合声明的类型要求。
@at.typecheck
# 注意力机制：将多个 Expert 的输入进行注意力计算
class Attention(nn.Module):

    # 模型的总配置参数序列（包括 lora 参数配置）
    # Sequence：序列类型，指一组有顺序、可以通过索引访问的元素，他可以是 list, str, ...
    configs: Sequence[Config]

    # Flax 提供的装饰器，允许在当前方法（通常是 __call__）中直接定义子模块
    @nn.compact
    # 执行函数
    def __call__(
        self,
        # xs：输入
        # xs 中的每个非空 x：[batch_size, seq_len_expert, width]
        xs, 
        # 每个 token 的位置：[batch_size, seq_len_sum]
        positions, 
        # 注意力掩码：[batch_size, 1, query_len, key_len]
        attn_mask, 
        # Key 和 Value 的缓存
        # kv_cache = (cache_k, cache_v)
        # cache_k：[batch_size, seq_len_cache, num_kv_heads, head_dim]
        # cache_v：[batch_size, seq_len_cache, num_kv_heads, head_dim]
        kv_cache
    ):

        # 1. 确保每个 Expert 的 头的特征维度 相同
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        # 2. 确保每个 Expert 的 Query 头数 相同
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        # 3. 确保每个 Expert 的 Key, Value 头数 相同
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        # 保存第一个非空输入的 type
        dtype = next(x.dtype for x in xs if x is not None)

        qkvs = []

        # 遍历每个 Expert 的 配置 以及 输入
        # zip(xs, self.configs, strict=True)：把 xs 和 self.configs 中相同位置的元素配对，zip() 返回的是迭代器，不是严格意义上的列表
        # strict=True：要求 zip() 中的所有序列长度相同；如果长度不同，就抛出 ValueError
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):

            # 若输入为空，则跳过本次循环，直接处理下一个 expert
            if x is None:
                continue

            # 若该 Expert 中 Query头的数量 = KeyValue头的数量
            # QKV的权重可以放在一起进行线性变换 + LoRA 微调
            if config.num_kv_heads == config.num_heads:

                # W_QKV 线性变换 + LoRA 微调
                qkv_einsum = lora.Einsum(
                    # W_QKV：[3, num_heads, width, head_dim]
                    # W_Q = W_QKV[0], W_K = W_QKV[1], W_V = W_QKV[2]
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    # W_QKV 的初始化器
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    # LoRA 配置参数
                    lora_config=config.lora_configs.get("attn"),
                )

                # x：[batch_size, seq_len_expert, width]
                # qkv_einsum("BSD,3KDH->3BSKH", x)：x 经过 W_QKV 线性变换 和 LoRA 微调后的结果
                # qkv_einsum("BSD,3KDH->3BSKH", x)：[3, batch_size, seq_len_expert, num_heads, head_dim]
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))

            else:

                # W_Q 线性变换 + LoRA 微调
                q_einsum = lora.Einsum(
                    # W_Q：[num_heads, width, head_dim]
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    # W_Q 的初始化器
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    # LoRA 配置参数
                    lora_config=config.lora_configs.get("attn"),
                )

                # x：[batch_size, seq_len_expert, width]
                # q：x 经过 W_Q 线性变换 和 LoRA 微调后的结果
                # q：[batch_size, seq_len_expert, num_heads, head_dim]
                q = q_einsum("BTD,NDH->BTNH", x)

                # W_KV 线性变换 + LoRA 微调
                kv_einsum = lora.Einsum(
                    # W_KV：[2, num_kv_heads, width, head_dim]
                    # W_K = W_KV[0], W_V = W_KV[1]
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    # W_KV 的初始化器
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    # LoRA 配置参数
                    lora_config=config.lora_configs.get("attn"),
                )

                # x：[batch_size, seq_len_expert, width]
                # k, v：x 经过 W_KV 线性变换 和 LoRA 微调后的结果
                # kv_einsum("BSD,2KDH->2BSKH", x)：[2, batch_size, seq_len_expert, num_kv_heads, head_dim]
                # k, v：[batch_size, seq_len_expert, num_kv_heads, head_dim]
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)

                # (q, k, v) = (
                #    q：[batch_size, seq_len_expert, num_heads, head_dim],
                #    k：[batch_size, seq_len_expert, num_kv_heads, head_dim],
                #    v：[batch_size, seq_len_expert, num_kv_heads, head_dim]
                #    且 num_heads != num_kv_heads
                # )
                qkvs.append((q, k, v))

        # 所有 expert 的 num_heads 和 num_kv_heads 配置一致，因此它们要么全部使用合并的 QKV 张量，要么全部分别保存 Q、K、V 张量

        # 方式一：所有 expert 都使用合并的 QKV 张量
        #
        # qkvs = [
        #     qkv1：[3, batch_size, seq_len_1, num_heads, head_dim]
        #     qkv2：[3, batch_size, seq_len_2, num_heads, head_dim]
        #     qkv3：[3, batch_size, seq_len_3, num_heads, head_dim]
        # ]
        #
        # zip(*qkvs) 按照第一个维度进行配对，得到：
        # (
        #     (qkv1[0], qkv2[0], qkv3[0]),  # 所有 expert 的 Q
        #     (qkv1[1], qkv2[1], qkv3[1]),  # 所有 expert 的 K
        #     (qkv1[2], qkv2[2], qkv3[2]),  # 所有 expert 的 V
        # )
        #
        # 其中：
        # qkv1[0].shape = [batch_size, seq_len_1, num_heads, head_dim]
        # qkv2[0].shape = [batch_size, seq_len_2, num_heads, head_dim]
        # qkv3[0].shape = [batch_size, seq_len_3, num_heads, head_dim]
        #
        # qkv1[1].shape = [batch_size, seq_len_1, num_heads, head_dim]
        # qkv2[1].shape = [batch_size, seq_len_2, num_heads, head_dim]
        # qkv3[1].shape = [batch_size, seq_len_3, num_heads, head_dim]
        #
        # qkv1[2].shape = [batch_size, seq_len_1, num_heads, head_dim]
        # qkv2[2].shape = [batch_size, seq_len_2, num_heads, head_dim]
        # qkv3[2].shape = [batch_size, seq_len_3, num_heads, head_dim]


        # 方式二：所有 expert 都使用 (q, k, v) 元组
        #
        # qkvs = [
        #     (q1, k1, v1),
        #     (q2, k2, v2),
        #     (q3, k3, v3),
        # ]
        #
        # zip(*qkvs) 按照元组中的位置进行配对，得到：
        # (
        #     (q1, q2, q3),  # 所有 expert 的 Q
        #     (k1, k2, k3),  # 所有 expert 的 K
        #     (v1, v2, v3),  # 所有 expert 的 V
        # )
        #
        # Q 的 shape：
        # q1.shape = [batch_size, seq_len_1, num_heads, head_dim]
        # q2.shape = [batch_size, seq_len_2, num_heads, head_dim]
        # q3.shape = [batch_size, seq_len_3, num_heads, head_dim]
        #
        # K 的 shape：
        # k1.shape = [batch_size, seq_len_1, num_kv_heads, head_dim]
        # k2.shape = [batch_size, seq_len_2, num_kv_heads, head_dim]
        # k3.shape = [batch_size, seq_len_3, num_kv_heads, head_dim]
        #
        # V 的 shape：
        # v1.shape = [batch_size, seq_len_1, num_kv_heads, head_dim]
        # v2.shape = [batch_size, seq_len_2, num_kv_heads, head_dim]
        # v3.shape = [batch_size, seq_len_3, num_kv_heads, head_dim]

        # q：[batch_size, seq_len_sum, num_heads, head_dim]
        # k：[batch_size, seq_len_sum, num_kv_heads, head_dim]
        # v：[batch_size, seq_len_sum, num_kv_heads, head_dim]
        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        # 对 Query（查询向量）应用旋转位置编码 RoPE
        # q：[batch_size, seq_len_sum, num_heads, head_dim]
        q = _apply_rope(q, positions=positions)

        # 在计算注意力分数之前缩放 Query，使点积结果不至于随着 head_dim 增大而过大
        # q = q / √head_dim
        q *= self.configs[0].head_dim ** -0.5

        # 对 Key（键向量）应用旋转位置编码 RoPE
        # k：[batch_size, seq_len_sum, num_kv_heads, head_dim]
        k = _apply_rope(k, positions=positions)

        # 确保 q, k, v 和 x 的数据类型相同
        assert q.dtype == k.dtype == v.dtype == dtype

        # 若 KV_Cache 不为空，则将 k_cache 和 v_cache 分别于 k, v 拼接
        if kv_cache is not None:

            # kv_cache = (cache_k, cache_v)
            # cache_k, cache_v：[batch_size, seq_len_cache, num_kv_heads, head_dim]
            cache_k, cache_v = kv_cache

            # cache_k, cache_v：[batch_size, seq_len_cache, num_kv_heads, head_dim]
            # 拼接前 k, v：[batch_size, seq_len_sum, num_kv_heads, head_dim]
            # 拼接后 k, v：[batch_size, seq_len_cache + seq_len_sum, num_kv_heads, head_dim]
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        # 三种注意力机制：
        # 1. MHA（Multi-Head Attention，多头注意力）：Q、K、V 的 Head 数量相同，每个 Q Head 对应一个 KV Head
        # 2. GQA（Grouped-Query Attention，分组查询注意力）：Q Head 数量大于 KV Head 数量，每组多个 Q Head 共享一个 KV Head，每组 Q Head 数量为 num_heads / num_kv_heads
        # 3. MQA（Multi-Query Attention，多查询注意力）：所有 Q Head 共享同一个 KV Head，即 K 和 V 各只有一个 Head
        # 当前 pi0 的模型配置采用 MQA（num_heads=8，num_kv_heads=1），但 Attention 代码同时支持 MHA、GQA 和 MQA

        # einops.rearrange() 的作用是按照指定的形状规则重新组织张量的维度，例如拆分维度、合并维度、调整维度顺序
        # (K G) -> K, G：将多个 Q Head 进行分组，每组有多个 Q Head 共享一个 KV Head 
        # K=self.configs[0].num_kv_heads：指定 KV Head 的大小，einops 会根据大小自动推断 G 的值
        # q：[(B)batch_size, (T)seq_len_sum, (K G)num_heads, (H)head_dim] ->
        #    [(B)batch_size, (T)seq_len_sum, (K)num_kv_heads, (G)num_q_heads_per_kv_heads, (H)head_dim]
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)

        # 计算注意力分数：
        # q：[(B)batch_size, (T)seq_len_sum, (K)num_kv_heads, (G)num_q_heads_per_kv_heads, (H)head_dim]
        # k：[(B)batch_size, (S)seq_len_cache + seq_len_sum, (K)num_kv_heads, (H)head_dim]
        # preferred_element_type=jnp.float32：指定 einsum 运算结果优先采用 float32 数据类型，具体的内部计算与累加精度由 JAX 后端决定
        # logits：[(B)batch_size, (K)num_kv_heads, (G)num_q_heads_per_kv_heads, (T)seq_len_sum, (S)seq_len_cache + seq_len_sum]
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        # 检查注意力掩码 attn_mask 的形状是否符合要求，如果不符合则抛出错误
        # 要求 attn_mask：[batch_size, 1, seq_len_sum, seq_len_cache + seq_len_sum]，是 token 之间的关系，与具体是哪个头无关
        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"注意力掩码实际的 shape：{attn_mask.shape}"
                f"要求的 shape：{(q.shape[0], 1, q.shape[1], k.shape[1])}"
            )

        # logits 数据类型能表示的有限最小值（代表屏蔽）
        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py

        # 掩码后的注意力分数
        # 将注意力掩码 attn_mask 为 False 处的注意力分数 logits 替换为 big_neg
        # attn_mask[:, :, None, :, :]：[batch_size, 1, 1, seq_len_sum, seq_len_cache + seq_len_sum]
        #
        # jnp.where(condition, x, y) 要求三个参数的 shape 能够广播到兼容的形状：
        # 所以 condition, x, y 都会广播成：[(B)batch_size, (K)num_kv_heads, (G)num_q_heads_per_kv_heads, (T)seq_len_sum, (S)seq_len_cache + seq_len_sum] 
        masked_logits = jnp.where(
            # 决定每个位置选择哪个值
            attn_mask[:, :, None, :, :], 
            # 条件为 True 时选择的值
            logits, 
            # 条件为 False 时选择的值
            big_neg
        )

        # 通过 softmax 层将 掩码后的注意力分数 变成 注意力权重
        # probs：[(B)batch_size, (K)num_kv_heads, (G)num_q_heads_per_kv_heads, (T)seq_len_sum, (S)seq_len_cache + seq_len_sum]
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        # 注意力输出（将 注意力权重 施加在 Value 上的结果）
        # probs：[(B)batch_size, (K)num_kv_heads, (G)num_q_heads_per_kv_heads, (T)seq_len_sum, (S)seq_len_cache + seq_len_sum]
        # v：[(B)batch_size, (S)seq_len_cache + seq_len_sum, (K)num_kv_heads, (H)head_dim]
        # encoded：[(B)batch_size, (T)seq_len_sum, (K)num_kv_heads, (G)num_q_heads_per_kv_heads, (H)head_dim]
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)

        # 注意力输出的中间结果
        # encoded：[(B)batch_size, (T)seq_len_sum, (KG)num_heads, (H)head_dim]
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        # 收集每个 Expert 的 注意力输出最终结果
        out = []

        # start, end：用于表示 Expert 的 seq_len 区间，[start, end-1]
        start = 0

        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    # W：[(N)num_heads, (H)head_dim, (D)width]
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )

                # 注意力输出的最终结果
                # encoded[:, start:end] = encoded[:, start: end, :, :]
                # encoded[:, start:end]：[(B)batch_size, (T)seq_len_expert, (N)num_heads, (H)head_dim]
                # W：[(N)num_heads, (H)head_dim, (D)width]
                # W_a：[(N)...] 要看具体 lora_config
                # W_b：[(N)...] 要看具体 lora_config
                # out_einsum("BTNH,NHD->BTD", encoded[:, start:end])：[(B)batch_size, (T)seq_len_expert, (D)width]
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))

                # 下一个 expert 的 seq_len 区间
                start = end

            else:
                out.append(None)

        # out 中每个非空元素：[(B)batch_size, (T)seq_len_expert, (D)width]
        # k, v：[batch_size, seq_len_cache + seq_len_sum, num_kv_heads, head_dim]
        return out, (k, v)

# @at.typecheck 是一个类型检查装饰器，用于检查函数调用是否符合声明的类型要求。
@at.typecheck
# 前馈神经网络：Gate + Linear
class FeedForward(nn.Module):

    # 输入特征维度，也是输出特征维度
    features: int

    # 隐藏层的特征维度
    hidden_dim: int

    # @nn.compact 允许在 __call__() 方法内部创建参数或子模块。
    # 例如 self.param() 和 nn.Dense() 都可以在该方法中使用。
    @nn.compact

    def __call__(self, x):

        # 保存输入数据的数据类型
        # x：[batch_size, seq_len, features]
        dtype = x.dtype  

        # W_Gate：[2, feaures, hidden_dim]
        w_gating = self.param(
            # W_Gate 的 名字
            "gating_einsum",
            # W_Gate 的 初始化器
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            # W_Gate 的 Shape
            (2, self.features, self.hidden_dim),
        ).astype(dtype) # 将 W_Gate 的数据类型改为 dtype，与输入数据 x 的数据类型相同

        # ff_gate：门控分支输出的中间结果
        # ff_gate：[batch_size, seq_len, hidden_dim]
        ff_gate = jnp.dot(x, w_gating[0])
        # gate_value：门控分支输出的最终结果
        # gate_value：[batch_size, seq_len, hidden_dim]
        gate_value = nn.gelu(ff_gate)

        # ff1：特征分支输出，与 gate_value 逐元素相乘，得到门控后的特征
        # ff1：[batch_size, seq_len, hidden_dim]
        ff1 = jnp.dot(x, w_gating[1])

        # activations：通过门控后的特征输出
        # activations：[batch_size, seq_len, hidden_dim]
        activations = gate_value * ff1

        # W_Linear：[hidden_dim, features]
        w_linear = self.param(
            # W_Linear 的 名字
            "linear",
            # W_Linear 的 初始化器
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            # W_Linear 的 shape
            (self.hidden_dim, self.features),
        ).astype(dtype) # 将 W_Linear 的数据类型改为 dtype，与输入数据 x 的数据类型相同

        # 最终的特征输出
        # outputs：[batch_size, seq_len, features]
        outputs = jnp.dot(activations, w_linear)

        assert outputs.dtype == dtype
    
        # outputs：[batch_size, seq_len, features]
        return outputs

# x：[batch_size, seq_len_expert, width] 
# y：[batch_size, seq_len_expert, width]
# gate：[batch_size, 1, width]
def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:

        # x + y：[batch_size, seq_len_expert, width] 
        return x + y
    
    # x + y * gate：[batch_size, seq_len_expert, width] 
    return x + y * gate

# @at.typecheck 是一个类型检查装饰器，用于检查函数调用是否符合声明的类型要求。
@at.typecheck
# Attention + FFN
class Block(nn.Module):

    # 由多个 Config 实例组成的元组，每个 Config 保存对应 Expert 的 模型配置
    configs: tuple[Config, ...]

    # Dropout Broadcast Dimensions（Dropout 广播维度）：指定 Dropout 掩码在哪些维度上共享，即这些维度上的元素使用相同的随机丢弃结果
    # 在掩码中：0 表示丢弃，1 表示保留
    # Eg：输入张量 shape = [B(3), S(4), W(5)]
    # B：batch_size，S：seq_len，W：width
    #
    # ():       每个元素独立随机判断，掩码 shape = (3, 4, 5)
    #           共有 B × S × W = 60 个独立判断位置，每个位置都有 dropout 的概率被丢弃
    # (0,):     不同样本的掩码相同，掩码 shape = (1, 4, 5)
    #           共有 S × W = 20 个独立判断位置，每个位置都有 dropout 的概率被丢弃
    # (1,):     同一样本中所有 token 的掩码相同，掩码 shape = (3, 1, 5)
    #           共有 B × W = 15 个独立判断位置，每个位置都有 dropout 的概率被丢弃
    # (2,):     同一个 token 的所有特征共享掩码，掩码 shape = (3, 4, 1)
    #           共有 B × S = 12 个独立判断位置，每个位置都有 dropout 的概率被丢弃
    # (0, 1):   所有样本、所有 token 的相同特征共享掩码，掩码 shape = (1, 1, 5)
    #           共有 W = 5 个独立判断位置，每个位置都有 dropout 的概率被丢弃
    # (0, 2):   不同样本中相同位置的 token 的所有特征共享掩码，掩码 shape = (1, 4, 1)
    #           共有 S = 4 个独立判断位置，每个位置都有 dropout 的概率被丢弃
    # (1, 2):   同一样本中所有 token 的所有特征共享掩码，掩码 shape = (3, 1, 1)
    #           共有 B = 3 个独立判断位置，每个位置都有 dropout 的概率被丢弃
    # (0, 1, 2):整个张量共享一个掩码，掩码 shape = (1, 1, 1)
    #           共有 1 个独立判断位置，该位置有 dropout 的概率被丢弃
    
    # dropout：每个独立判断位置被丢弃的概率
    dropout: float = 0.0

    # dropout_bdims:指定哪些维度共享同一个随机丢弃结果
    dropout_bdims: tuple[int, ...] = ()

    # @nn.compact 允许在 __call__() 方法内部创建参数或子模块。
    # 例如 self.param() 和 nn.Dense() 都可以在该方法中使用。
    @nn.compact

    def __call__(
        self,
        # 多个 Expert 输入的集合
        # 每个非空 x：[batch_size, seq_len_expert, width] 
        xs, 
        # Key 和 Value 的缓存
        # kv_cache = (cache_k, cache_v)
        # cache_k：[batch_size, seq_len_cache, num_kv_heads, head_dim]
        # cache_v：[batch_size, seq_len_cache, num_kv_heads, head_dim]
        kv_cache, 
        # 所有非空 Expert 的 token 位置编号
        # positions：[batch_size, seq_len_sum]
        # seq_len_sum：所有非空 Expert 的 token 数量之和
        positions, 
        # 注意力掩码：控制 Query Token 可以关注哪些 Key Token，与头无关
        # attn_mask：[batch_size, 1, query_len, key_len]
        attn_mask, 
        # AdaRMSNorm 的条件输入
        # adarms_cond：每个 Expert 对应一个条件张量或 None
        # adarms_cond[i]：[batch_size, cond_dim] 或 None
        # 当前 OpenPI 调用中，cond_dim 通常对应当前 Expert 的 width
        adarms_cond, 
        # 是否采用确定性模式，主要控制 Dropout 是否启用随机丢弃
        # True：关闭 Dropout 的随机丢弃（通常用于推理）
        # False：启用 Dropout 的随机丢弃（通常用于训练）
        deterministic=True
    ): 

        #########################################################################
        #
        # 一、 Attention 残差连接（gate_attn 来自 RMSNorm）
        # 
        # 1. x_attn = Dropout(Attention(RMSNorm(x)))
        #
        # 2. x_attn_mixed = 
        #   1) x + x_attn                , gate_attn  = None
        #   2) x + gate_attn * x_attn    , gate_attn != None
        #
        #########################################################################

        # 对 xs 中的输入张量施加分片约束（Sharding Constraint），指定张量在设备上的分布方式，不改变张量的逻辑 shape
        xs = sharding.activation_sharding_constraint(xs)

        # drop 模块（self.dropout == 0.0 时 被视为 False，触发恒等映射函数）
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        # Attention 模块
        # self.configs：各个 Expert 的模型配置
        attn = Attention(configs=self.configs, name="attn")

        # pre_attn：保存各个 Expert 的输入经过 方均根归一化(RMSNorm)后 和在 Attention 之前的归一化结果
        # 每个非空元素：[batch_size, seq_len_expert, width]
        pre_attn = []

        # gates：保存各个 Expert 对应的门控参数
        # 每个元素可能为 [batch_size, 1, width] 或 None
        gates = []

        # 遍历每个 Expert 的输入，并进行归一化
        for i, x in enumerate(xs):

            # 如果 x 非空
            if x is not None:

                # x：[batch_size, seq_len_expert, width]
                # gate：[batch_size, 1, width] 或 None（如果 adarms_cond[i] is None，那么 gate=None）
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])

            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        # 对 pre_attn 中的输入张量施加分片约束（Sharding Constraint），指定张量在设备上的分布方式，不改变张量的逻辑 shape
        pre_attn = sharding.activation_sharding_constraint(pre_attn)

        # post_attn 中每个非空元素：[(B)batch_size, (T)seq_len_expert, (D)width]
        # kv_cache = (k, v)
        # k, v：[batch_size, seq_len_cache + seq_len_sum, num_kv_heads, head_dim]
        post_attn, kv_cache = attn(
            pre_attn,  # 非空元素为：[batch_size, seq_len_expert, width]
            positions, # shape = [batch_size, seq_len_sum]
            attn_mask, # shape = [batch_size, 1, query_len, key_len]
            kv_cache   # kv_cache[0], kv_cache[1]：[batch_size, seq_len_cache, num_kv_heads, head_dim]
        )

        # 遍历 post_attn 这棵 PyTree 结构，对其中的叶子节点逐个应用 Dropout
        # 每个非 None 元素的输入和输出 shape 不变：[batch_size, seq_len_expert, width]
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)

        # 对 post_attn 中的输入张量施加分片约束（Sharding Constraint），指定张量在设备上的分布方式，不改变张量的逻辑 shape
        post_attn = sharding.activation_sharding_constraint (post_attn)

        # 带门控的 Attention 残差连接
        # 
        # xs 的每个非空元素 x：[batch_size, seq_len_expert, width] 
        # post_attn 的每个非空元素 y：[batch_size, seq_len_expert, width]
        # gates 的每个非空元素 gate：[batch_size, 1, width]
        #
        # 结果：xs 的每个非空元素 x：[batch_size, seq_len_expert, width] 
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]

        #########################################################################
        #
        # 二、 FeedForward 残差连接（gate_ffn 来自 RMSNorm）
        # 
        # 1. x_ffn = Dropout(FeedForward(RMSNorm(x_attn_mixed)))
        #
        # 2. x_end = 
        #   1) x_attn_mixed + x_ffn              , gate_ffn  = None
        #   2) x_attn_mixed + gate_ffn * x_ffn   , gate_ffn != None
        #
        #########################################################################

        # 对 xs 中的输入张量施加分片约束（Sharding Constraint），指定张量在设备上的分布方式，不改变张量的逻辑 shape
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []

        # RMSNorm + 门控 + Lora微调
        # xs 的每个非空元素 x：[batch_size, seq_len_expert, width] 
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):

            # 若 x 非空
            if x is not None:
                
                # 方均根归一化
                # x：[batch_size, seq_len_expert, width]
                # gate：[batch_size, 1, width] 或 None（如果 adarms_cond[i] is None，那么 gate=None）
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  

                # 门控 + LoRA微调
                # x：[batch_size, seq_len_expert, width]
                x = lora.FeedForward( 
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)

            
            out.append(x)
            gates.append(gate if x is not None else None)

        # 对 out 中的输入张量施加分片约束（Sharding Constraint），指定张量在设备上的分布方式，不改变张量的逻辑 shape
        out = sharding.activation_sharding_constraint(out)

        # 遍历 out 这棵 PyTree 结构，对其中的叶子节点逐个应用 Dropout
        # 每个非 None 元素的输入和输出 shape 不变：[batch_size, seq_len_expert, width] 
        out = jax.tree.map(lambda x: drop(x, deterministic), out)

        # 带门控的 FFN 残差连接
        # 
        # xs 的每个非空元素 x：[batch_size, seq_len_expert, width] 
        # out 的每个非空元素 y：[batch_size, seq_len_expert, width]
        # gates 的每个非空元素 gate：[batch_size, 1, width]
        #
        # 结果：xs 的每个非空元素 x：[batch_size, seq_len_expert, width]  
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]

        # 对 xs 中的输入张量施加分片约束（Sharding Constraint），指定张量在设备上的分布方式，不改变张量的逻辑 shape
        xs = sharding.activation_sharding_constraint(xs)

        # xs 的每个非空元素 x：[batch_size, seq_len_expert, width] 
        # kv_cache = (k, v)
        # k, v：[batch_size, seq_len_cache + seq_len_sum, num_kv_heads, head_dim]
        return xs, kv_cache

# TypeAlias：明确告诉类型检查工具，KVCache 是一个类型别名
# KVCache：保存所有 Transformer Block 的 Key 和 Value 缓存
# cache_k：[depth, batch_size, seq_len_cache, num_kv_heads, head_dim]
# cache_v：[depth, batch_size, seq_len_cache, num_kv_heads, head_dim]
# depth：Transformer Block 的层数
KVCache: TypeAlias = tuple[
    at.Float[at.Array, "l b _t _k _h"], 
    at.Float[at.Array, "l b _t _v _h"]
]

# @at.typecheck 是一个类型检查装饰器，用于检查函数调用是否符合声明的类型要求。
@at.typecheck
class Module(nn.Module):

    # 模型的总配置参数序列（包括 lora 参数配置）
    # Sequence：序列类型，指一组有顺序、可以通过索引访问的元素，他可以是 list, str, ...
    configs: Sequence[Config] # [Config("gemma_2b"), Config("gemma_300m")]

    # 词嵌入结果的指定类型
    embed_dtype: str # "bfloat16"

    # dropout：每个独立判断位置被丢弃的概率
    dropout: float = 0.0

    # dropout_bdims:指定哪些维度共享同一个随机丢弃结果
    dropout_bdims: tuple[int, ...] = ()

    # adarms：是否使用 Adaptive RMSNorm 的配置标志
    adarms: bool = False

    # 定义子模块
    def setup(self):

        # 确保所有 Expert 的 Transformer Block 层数相同
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        # 词嵌入模块 Embedder
        self.embedder = Embedder(
            # PaliGemma 的词表大小
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            # 使用第一个 Expert 的隐藏特征维度作为词嵌入维度
            embed_dim=self.configs[0].width,  
            # 子模块名称
            name="embedder",
        )

        # 使用 nn.remat 包装 Transformer Block
        # 作用：在反向传播时重新计算部分前向结果，减少训练过程中需要保存的中间激活值，从而降低显存占用
        # 代价：需要额外执行部分前向计算，增加计算开销
        block_cls = nn.remat(
            # 被包装的 Transformer Block 类
            Block,
            # prevent_cse=False：不要求额外阻止 Common Subexpression Elimination（公共子表达式消除）
            prevent_cse=False,
            # static_argnums=(5,)：将第 6 个位置参数 deterministic 视为静态参数
            # 这里的参数索引从 0 开始，不包含 self
            # 0：xs
            # 1：kv_cache
            # 2：positions
            # 3：attn_mask
            # 4：adarms_cond
            # 5：deterministic
            static_argnums=(5,),
            # nothing_saveable：指定重计算策略，尽量不保存可重计算的中间激活值，在反向传播时重新计算这些结果。
            policy=jax.checkpoint_policies.nothing_saveable,
        )

        # 使用 nn.scan 创建多层 Transformer Block
        # nn.scan：Flax 提供的循环变换，用于沿指定的扫描维度，重复执行同一个模块结构
        # 此处将 Block 重复执行 depth 次，构成完整的 Transformer   
        # 每一层：
        #   1. 接收上一层输出的 xs
        #   2. 接收对应层的 KV Cache
        #   3. 执行 Attention 和 FeedForward
        #   4. 输出新的 xs 和 KV Cache
        # xs 会逐层传递，而每层使用自己的参数
        # 
        # nn.scan(...)：返回经过 nn.scan 变换的 Block 模块类（scanned_block_cls）
        self.layers = nn.scan(
            # 已经经过 nn.remat 包装的 Transformer Block
            block_cls,
            # 将所有 Transformer Block 的参数沿第 0 维（depth）堆叠
            # 每个 Transformer Block 拥有独立的可学习参数
            # nn.scan 执行时，每一层使用对应的参数
            variable_axes={"params": 0},
            # 指定随机数生成器是否在不同扫描层之间拆分
            split_rngs={
                # 不同 Transformer Block 使用独立的参数初始化随机数
                "params": True, 
                # 不同 Transformer Block 使用独立的 Dropout 随机数
                "dropout": True
            },
            # 指定哪些输入参数沿 Transformer 层维度拆分，哪些输入参数在所有层之间共享
            in_axes=(
                              # xs           ：xs 是 nn.scan 的 carry（循环状态），它不包含在 in_axes 中
                0,            # kv_cache     ：沿 depth 维（0 维）度拆分，每层使用对应的 KV Cache
                nn.broadcast, # positions    ：所有 Transformer Block 共享
                nn.broadcast, # mask         ：所有 Transformer Block 共享
                nn.broadcast, # adarms_cond  ：所有 Transformer Block 共享
                nn.broadcast, # deterministic：所有 Transformer Block 共享
            ),  
            # 指定执行多少层 Transformer Block
            length=self.configs[0].depth,
        )(
            # 由多个 Config 实例组成的元组，每个 Config 保存对应 Expert 的 模型配置
            configs=self.configs,
            # dropout：每个独立判断位置被丢弃的概率
            dropout=self.dropout,
            # dropout_bdims:指定哪些维度共享同一个随机丢弃结果
            dropout_bdims=self.dropout_bdims,
        )  

        # 为每个 Expert 模块创建一个独立的 RMSNorm 模块，并将所有的 RMSNorm 模块保存在列表中
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    # @at.typecheck 是一个类型检查装饰器，用于检查函数调用是否符合声明的类型要求。
    @at.typecheck
    # tokens：[batch_size, seq_len]
    # return：[batch_size, seq_len, embed_dim]
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    # @at.typecheck 是一个类型检查装饰器，用于检查函数调用是否符合声明的类型要求。
    @at.typecheck
    def __call__(
        self,
        # 所有 Expert 的输入 Embedding 序列
        # embedded[i]：[batch_size, seq_len_expert, width] | None
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        # 所有参与当前计算的 Token 的位置索引，用于 RoPE（旋转位置编码）
        # positions：[batch_size, seq_len_sum]
        positions: at.Int[at.Array, "b t"],
        # Attention Mask（注意力掩码），控制 Query 能否关注对应的 Key
        #    True：允许当前 Query 关注对应的 Key
        #    False：禁止当前 Query 关注对应的 Key
        # mask：[batch_size, query_len, key_len]
        mask: at.Bool[at.Array, "b t s"],
        # AdaRMSNorm（Adaptive RMS Normalization，自适应均方根归一化）的条件输入
        # adarms_cond[i]：[batch_size, cond_dim] | None
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        # 前侧参数位置传参，关键字传参都可以，后续参数必须通过关键字传入，不能通过位置传入
        *, 
        # Key-Value Cache（键值缓存），用于保存 Transformer Attention 的 K、V
        # kv_cache = (k_cache, v_cache)
        # k, v：[depth, batch_size, seq_len_cache, num_kv_heads, head_dim]
        kv_cache: KVCache | None = None,
        # 是否采用确定性模式，主要控制 Dropout 是否启用随机丢弃
        # True：关闭 Dropout 的随机丢弃（通常用于推理）
        # False：启用 Dropout 的随机丢弃（通常用于训练）
        deterministic: bool = True,
    ) -> tuple[
            # 所有 Expert 的最终隐藏特征
            # [batch_size, seq_len_expert, width] | None
            Sequence[at.Float[at.Array, "b _t _d"] | None], 
            # 更新后的所有 Transformer Block 的 KV Cache
            # kv_cache = (k_cache, v_cache)
            # k, v：[depth, batch_size, seq_len_cache, num_kv_heads, head_dim]
            KVCache
        ]:

        # 遍历 post_attn 这棵 PyTree 结构，对其中的叶子节点逐个统一 Expert 的输入 Embedding 的数据类型
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)

        # mask：[batch_size, 1, query_len, key_len]
        mask = jnp.asarray(mask)[:, None, :, :]

        # 如果不使用 AdaRMSNorm，给每一个 Expert 创建一个空的 adarms_cond：[None, None, ...]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        # 执行 self.layers 中的所有 Transformer Block（共 depth 层）。
        # 每层的 embedded 和 kv_cache 会传递给下一层。
        # self.layers 是通过 nn.scan 创建的多层 Transformer
    
        # 1. 
        # embedded：经过 depth 层 Transformer Block 后，所有 Expert 的最终隐藏特征。
        # Sequence：每个元素对应一个 Expert。
        # 每个非空元素 Shape：[batch_size, seq_len_expert, width_expert]
        # None：对应 Expert 未参与计算。
        # 例如：embedded = [vlm_hidden_states, action_hidden_states]
        # 注意：这里的 embedded 尚未经过 Module 最后的 Final RMSNorm。
        #
        # 2. 
        # kv_cache：所有 Transformer Block 返回的 Key、Value 缓存。
        # KVCache = (key_cache, value_cache)
        # key_cache Shape：[depth, batch_size, seq_len_cache, num_kv_heads, head_dim]
        # value_cache Shape：[depth, batch_size, seq_len_cache, num_kv_heads, head_dim]
        # depth：Transformer Block 的层数。
        # seq_len_cache：缓存中的 Token 数量。
        # num_kv_heads：Key、Value 的 Attention Head 数量。
        # head_dim：每个 Attention Head 的特征维度。
        # 这些缓存可在后续 Attention 计算中复用，避免重复计算历史 Token 的 K、V。
        embedded, kv_cache = self.layers(
            # embedded：所有 Expert 当前的隐藏特征。
            # 每个非空元素 Shape：[batch_size, seq_len_expert, width_expert]
            embedded,

            # kv_cache：所有 Transformer Block 的 Key、Value 缓存。
            # None：没有传入已有的 KV Cache。
            # 非空时，每层使用对应的 KV Cache。
            kv_cache,

            # positions：所有 Token 的位置索引，用于 RoPE。
            # Shape：[batch_size, seq_len_sum]
            # 所有 Transformer Block 共享。
            positions,

            # mask：Attention Mask，控制 Query 能否关注对应的 Key。
            # Shape：[batch_size, 1, query_len, key_len]
            # 所有 Transformer Block 共享。
            mask,

            # adarms_cond：每个 Expert 的 AdaRMSNorm 条件输入。
            # 每个元素 Shape：[batch_size, cond_dim] 或 None。
            # 所有 Transformer Block 共享。
            adarms_cond,

            # deterministic：控制 Dropout 是否执行随机丢弃。
            # True：关闭 Dropout；False：允许按配置的概率执行 Dropout。
            # 所有 Transformer Block 共享。
            deterministic,
        )

        # 确保所有非空的 Exper 隐藏层 类型相同
        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        # f(e, a)[0]  # normalized_embedded
        # f(e, a)[1]  # gate
        # f(e, a)[0]中每个非空元素：[batch_size, seq_len_expert, width_expert]
        #
        # KVCache = (key_cache, value_cache)
        # key_cache Shape：[depth, batch_size, seq_len_cache, num_kv_heads, head_dim]
        # value_cache Shape：[depth, batch_size, seq_len_cache, num_kv_heads, head_dim]
        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

  
    # 自定义方法
    def init(
        self, 
        # use_adarms：一个布尔值序列，表示每个 Expert 是否需要初始化 AdaRMSNorm 分支
        use_adarms: Sequence[bool]
    ):

        # 在 Flax Linen 中，setup() 通常在模块首次被实际使用时，由 Flax 自动执行。
        # 单纯实例化 Embedder(...) 不会立即触发 Embedder.setup()，因此，在 Flax 参数初始化上下文中调用 self.embed()，可以触发 Embedder.setup()，从而初始化 input_embedding_table（词嵌入表参数）
        # self.embed() 返回的 Embedding 不需要使用，直接丢弃
        # jnp.zeros((1, 1), dtype=jnp.int32) = [[0]]：[batch_size(1), seq_len(1)]
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))

        # 在 Python 中，self(...) 相当于调用当前对象的 __call__() 方法
        
        # 调用 Module.__call__()，通过虚拟输入触发 setup(), 从而触发 Transformer 各子模块的参数初始化。
        # 每个 Expert 只使用一个虚拟 Token，所有输入均为全零张量。
        # 返回的 embedded 和 kv_cache 不需要使用。
        self(
            # embedded = [
            #   Expert_1：[batch_size=1, seq_len_expert=1, width_1],
            #   Expert_2：[batch_size=1, seq_len_expert=1, width_2],
            #   ...
            # ]
            [jnp.zeros((1, 1, c.width)) for c in self.configs],

            # positions：[batch_size=1, seq_len_sum=num_experts]
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),

            # mask：[batch_size=1, query_len=num_experts, key_len=num_experts]
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),

            # adarms_cond = [
            #   Expert_1：[batch_size=1, cond_dim=width_1] 或 None,
            #   Expert_2：[batch_size=1, cond_dim=width_2] 或 None,
            #   ...
            # ]
            # use_adarms：每个 Expert 是否使用 AdaRMSNorm 条件输入。
            # u=True：创建对应 Expert 的全零条件张量。
            # u=False：使用 None，不提供条件输入。
            # c.width：当前 Expert 的隐藏特征维度。
            # strict=True：要求 use_adarms 与 self.configs 的长度一致。
            adarms_cond=[
                jnp.zeros((1, c.width)) if u else None
                for u, c in zip(use_adarms, self.configs, strict=True)
            ],
        )
