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

"""A refactored and simplified ViT adoptation for Pi, taken from big_vision."""

from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.sharding as sharding

# 生成二维正弦余弦位置编码（2D Sinusoidal Positional Embedding）
def posemb_sincos_2d(
    # patch_rows：patch 的行数
    h, 
    # patch_cols：patch 的列数
    w, 
    # 每个 patch token 的特征维度：features
    width, 
    # 频率缩放基数
    temperature=10_000.0, 
    # 位置编码的数据类型
    dtype=jnp.float32
):
    # h = patch_rows：Patch 的行数
    # w = patch_cols：Patch 的列数
    #
    # jnp.mgrid[:h, :w]：生成每个 Patch 的行坐标和列坐标
    #
    # 例如：h = 2，w = 3
    # jnp.mgrid[:2, :3]：
    #   :2 = [0, 1]     # 行坐标范围
    #   :3 = [0, 1, 2]  # 列坐标范围
    #
    # 对应的 Patch 坐标：
    # (0, 0) (0, 1) (0, 2)
    # (1, 0) (1, 1) (1, 2)
    #
    # y：记录每个 Patch 的行坐标
    # y = [[0, 0, 0],
    #      [1, 1, 1]]
    #
    # x：记录每个 Patch 的列坐标
    # x = [[0, 1, 2],
    #      [0, 1, 2]]
    #
    # jnp.mgrid[:h, :w].shape = (2, patch_rows, patch_cols)
    # y.shape = x.shape = (patch_rows, patch_cols)
    #
    # 例如：y.shape = x.shape = (2, 3)
    #
    # y, x：[patch_rows, patch_cols]
    y, x = jnp.mgrid[:h, :w]

    # width 要是 4 的倍数
    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    
    # omega：不同特征维度对应的角频率（Angular Frequency）
    # omega = [0, 1/(width//4-1), ..., (width//4-2)/(width//4-1), 1]
    # omega = [width//4]
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    # omega = [1/(10000^(0)), ..., 1/(10000^1)]
    omega = 1.0 / (temperature**omega)

    # y：[patch_rows, patch_cols]
    # y.flatten()：[patch_rows * patch_cols]
    # omega：[width // 4]
    #
    # "m,d->md"：向量外积（Outer Product）
    #   将 y.flatten() 中的每个元素分别与 omega 中的所有元素相乘。
    #   m：Patch 索引，共 patch_rows * patch_cols 个
    #   d：角频率索引，共 width // 4 个
    #
    # 计算公式：
    #   output[m, d] = y.flatten()[m] * omega[d]
    #
    # 结果 y：[patch_rows * patch_cols, width // 4]
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    
    # 结果 x：[patch_rows * patch_cols, width // 4]
    x = jnp.einsum("m,d->md", x.flatten(), omega)

    # jnp.sin(x)：[patch_rows * patch_cols, width // 4]
    # jnp.cos(x)：[patch_rows * patch_cols, width // 4]
    # jnp.sin(y)：[patch_rows * patch_cols, width // 4]
    # jnp.cos(y)：[patch_rows * patch_cols, width // 4]
    # pe：[patch_rows * patch_cols, width]
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)

    # pe：[1, patch_rows * patch_cols, width]
    return jnp.asarray(pe, dtype)[None, :, :]

# 获取位置编码：Get Positinal Embedding
def get_posemb(
    self, 
    # 位置编码类型：
    # 1. "learn"：可学习位置编码（Learnable Positional Embedding）
    # 2. "sincos2d"：二维正弦余弦位置编码（2D Sinusoidal Positional Embedding）
    typ, 
    # patch 排列方式：[patch_rows, patch_cols]
    seqshape, 
    # 每个 patch token 的特征维度：features
    width, 
    # 可学习位置编码参数的名称
    name, 
    # 位置编码的数据类型
    dtype=jnp.float32
):
    # 1. "learn"：可学习位置编码（Learnable Positional Embedding）
    if typ == "learn":
        # param：[1, patch_rows * patch_cols, width]
        return self.param(
            # 1. 参数名称
            name,
            # 2. 初始化器
            # stddev（standard deviation 标准差）= 1 / √features
            nn.initializers.normal(stddev=1 / np.sqrt(width)),
            # 3. shape：[1, patch_rows * patch_cols, width]
            # prod（product）：乘积
            (1, np.prod(seqshape), width),
            # 4. 位置编码的数据类型
            dtype,
        )
    # 2. "sincos2d"：二维正弦余弦位置编码（2D Sinusoidal Positional Embedding）
    if typ == "sincos2d":
        # posemb_sincos_2d：[1, patch_rows * patch_cols, width]
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)

    raise ValueError(f"Unknown posemb type: {typ}")

class MlpBlock(nn.Module):
    """Transformer MLP / Feed-Forward Block（多层感知机 / 前馈网络）"""

    # mlp_dim：MLP 中间隐藏层的特征维度
    # None：默认使用输入特征维度 features 的 4 倍
    # 例如：features=768，mlp_dim=None → 3072
    # Shape：[batch_size, seq_len, 768] → [batch_size, seq_len, 3072] → [batch_size, seq_len, 768]
    mlp_dim: int | None = None

    # dropout：Dropout 概率
    # 0.1：训练时每个元素有 10% 的概率被置为 0
    # 0.0：不进行随机丢弃
    # Dropout 不改变 Shape
    dropout: float = 0.0

    # dtype_mm：矩阵运算的数据类型（Matrix Multiplication Dtype）
    # "float32"：32 位浮点数
    # "float16"：16 位浮点数
    # "bfloat16"：16 位 Brain Floating Point
    dtype_mm: str = "float32"

    # @nn.compact 允许在 __call__() 方法内部创建参数或子模块
    # 例如 self.param() 和 nn.Dense() 都可以在该方法中使用
    @nn.compact
    def __call__(
        # self：当前 MlpBlock 实例
        self,

        # x：输入 Token 特征
        # Shape：[batch_size, seq_len, features]
        # 例如：[2, 197, 768]
        x,

        # deterministic：是否使用确定性模式
        # True：关闭 Dropout，通常用于推理
        # False：允许启用 Dropout，通常用于训练
        deterministic=True,
    ):  # noqa: FBT002
        """执行 Transformer MLP Block 的前向传播"""

        # inits：保存 Dense 层参数初始化方法的字典
        inits = {
            # kernel_init：权重矩阵（Weight Matrix）的初始化方法
            # xavier_uniform()：Xavier Uniform Initialization（Xavier 均匀初始化）
            # 对二维权重 [input_dim, output_dim]：
            #   fan_in = input_dim
            #   fan_out = output_dim
            #   初始化范围为 ±sqrt(6 / (fan_in + fan_out))
            "kernel_init": nn.initializers.xavier_uniform(),

            # bias_init：偏置向量（Bias Vector）的初始化方法
            # normal()：使用正态分布随机初始化
            "bias_init": nn.initializers.normal(
                # stddev：正态分布的标准差（Standard Deviation）
                # 1e-6 = 0.000001，均值默认为 0
                stddev=1e-6,
            ),
        }

        # x.shape：[batch_size, seq_len, features]
        # 第一个 _：batch_size，此处不使用
        # 第二个 _：seq_len，此处不使用
        # d：features，输入 Token 的特征维度
        # 例如：x.shape=[2, 197, 768] → d=768
        _, _, d = x.shape

        # ============================================================
        # 1. 第一层全连接层（Fully Connected Layer）
        # ============================================================

        # nn.Dense：对输入最后一个特征维度执行线性变换
        # 公式：output = x @ kernel + bias
        #
        # 输入 x：[batch_size, seq_len, features]
        # 权重 kernel：[features, mlp_dim]
        # 偏置 bias：[mlp_dim]
        # 输出 x：[batch_size, seq_len, mlp_dim]
        #
        # 例如：
        # x：[2, 197, 768]
        # kernel：[768, 3072]
        # bias：[3072]
        # 输出：[2, 197, 3072]
        x = nn.Dense(
            # features：Dense 层的输出特征维度
            # self.mlp_dim 非零且非 None 时使用指定值，否则使用 4*d
            # 例如：d=768，self.mlp_dim=None → 3072
            self.mlp_dim or 4 * d,

            # dtype：Dense 层计算使用的数据类型
            dtype=self.dtype_mm,

            # **inits：展开初始化参数字典
            # *inits：只解开键
            # **inits：解开键值对
            # 等价于分别传入 kernel_init 和 bias_init
            **inits,
        )(
            # inputs：输入 Token 特征
            # Shape：[batch_size, seq_len, features]
            x,
        )

        # ============================================================
        # 2. GELU 激活函数
        # ============================================================

        # GELU：Gaussian Error Linear Unit（高斯误差线性单元）
        # 公式：GELU(x) = x * Φ(x)
        # Φ(x)：标准正态分布的累积分布函数
        # 作用：引入非线性变换，不改变 Shape
        x = nn.gelu(
            # x：第一层 Dense 的输出
            # Shape：[batch_size, seq_len, mlp_dim]
            x,
        )

        # ============================================================
        # 3. Dropout
        # ============================================================

        # Dropout：训练时随机丢弃部分隐藏层特征
        # 例如：dropout=0.1，保留的元素按 1/(1-0.1) 缩放
        # 输入/输出：[batch_size, seq_len, mlp_dim]
        x = nn.Dropout(
            # rate：随机丢弃概率
            rate=self.dropout,
        )(
            # inputs：需要执行 Dropout 的输入张量
            # Shape：[batch_size, seq_len, mlp_dim]
            x,

            # deterministic：是否关闭 Dropout
            # True：关闭；False：启用（当 rate>0）
            deterministic,
        )

        # ============================================================
        # 4. 第二层全连接层（Fully Connected Layer）
        # ============================================================

        # 将隐藏层特征维度恢复为原始 features
        # 公式：output = x @ kernel + bias
        #
        # 输入 x：[batch_size, seq_len, mlp_dim]
        # 权重 kernel：[mlp_dim, features]
        # 偏置 bias：[features]
        # 输出：[batch_size, seq_len, features]
        #
        # 例如：
        # x：[2, 197, 3072]
        # kernel：[3072, 768]
        # bias：[768]
        # 输出：[2, 197, 768]
        #
        # 注意：此处不再执行 Dropout
        # 外层 Encoder1DBlock 会对 MLP 输出再次执行 Dropout
        return nn.Dense(
            # features：输出特征维度，恢复为输入的 d
            d,

            # dtype：Dense 层计算使用的数据类型
            dtype=self.dtype_mm,

            # **inits：权重和偏置的初始化方法
            **inits,
        )(
            # inputs：经过 GELU 和 Dropout 的隐藏层特征
            # Shape：[batch_size, seq_len, mlp_dim]
            x,
        )

class Encoder1DBlock(nn.Module):
    """单层 Transformer Encoder Block（Multi-Head Self-Attention + MLP）"""

    # mlp_dim：多层感知机（Multi-Layer Perceptron）的隐藏层维度
    # None：默认使用输入特征维度 features 的 4 倍（由 MlpBlock 内部实现）
    #
    # 例如：
    #   features = 768
    #   mlp_dim = None → 3072
    #
    # MLP 内部典型 Shape 变化：
    #   [batch_size, seq_len, 768]
    #       ↓
    #   [batch_size, seq_len, 3072]
    #       ↓
    #   [batch_size, seq_len, 768]
    mlp_dim: int | None = None

    # num_heads：多头注意力（Multi-Head Attention）的注意力头数量
    #
    # 例如：
    #   features = 768
    #   num_heads = 12
    #   head_dim = 768 // 12 = 64
    #
    # Q、K、V 的典型 Shape：
    #   [batch_size, seq_len, num_heads, head_dim]
    num_heads: int = 12

    # dropout：Dropout 概率
    #
    # 例如：
    #   dropout = 0.1：训练时随机将部分元素置为 0
    #   dropout = 0.0：不进行随机丢弃
    #
    # Dropout 不改变 Shape
    dropout: float = 0.0

    # dtype_mm：矩阵运算使用的数据类型（Matrix Multiplication Dtype）
    #
    # 例如：
    #   "float32"：32 位浮点数
    #   "float16"：16 位浮点数
    #   "bfloat16"：16 位 Brain Floating Point
    #
    # 控制下方 LayerNorm、Attention、MLP 中相关计算的数据类型
    # 具体计算精度由各子模块实现决定
    dtype_mm: str = "float32"

    # @nn.compact：
    #   Flax Linen 装饰器
    #   允许在 __call__() 中直接创建并调用子模块
    #   例如 nn.LayerNorm、nn.MultiHeadDotProductAttention、MlpBlock
    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002

        # x：输入 Token 特征
        # Shape：[batch_size, seq_len, features]
        #
        # batch_size：批量大小
        # seq_len：Token 序列长度
        # features：每个 Token 的特征维度
        #
        # 例如：
        #   x：[2, 197, 768]
        #
        # deterministic：是否使用确定性模式
        #   True：关闭 Dropout，通常用于推理
        #   False：启用 Dropout，通常用于训练

        # out：保存 Transformer Block 的中间计算结果
        #
        # 后续保存：
        #   out["sa"]：Self-Attention 的输出
        #   out["+sa"]：第一次残差连接后的输出
        #   out["mlp"]：MLP 的输出
        #   out["+mlp"]：第二次残差连接后的输出
        out = {}

        # activation_sharding_constraint()：
        #   对激活张量施加设备分片约束（Activation Sharding Constraint）
        #
        # 作用：
        #   指定或约束 x 在多个计算设备上的分布方式
        #   具体分片规则由 sharding 模块内部实现决定
        #
        # 注意：
        #   不改变 x 的逻辑 Shape
        #
        # x：[batch_size, seq_len, features]
        x = sharding.activation_sharding_constraint(x)

        # ============================================================
        # 1. Multi-Head Self-Attention（多头自注意力）
        # ============================================================

        # nn.LayerNorm()：层归一化（Layer Normalization）
        #
        # dtype=self.dtype_mm：
        #   指定 LayerNorm 输出及相关计算使用的数据类型
        #
        # 默认对最后一个特征维度 features 进行归一化
        #
        # 输入 x：[batch_size, seq_len, features]
        # 输出 y：[batch_size, seq_len, features]
        #
        # 例如：
        #   x：[2, 197, 768]
        #   y：[2, 197, 768]
        #
        # LayerNorm 不改变 Shape
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)

        # nn.MultiHeadDotProductAttention()：
        #   多头点积注意力（Multi-Head Dot-Product Attention）
        #
        # num_heads=self.num_heads：
        #   Attention Head 的数量
        #
        # kernel_init=nn.initializers.xavier_uniform()：
        #   使用 Xavier Uniform 初始化注意力模块中的权重矩阵
        #
        # deterministic=deterministic：
        #   控制注意力模块内部的 Dropout 是否启用
        #
        # dtype=self.dtype_mm：
        #   Attention 模块计算使用的数据类型
        #
        # (y, y)：
        #   第一个 y：Query 的输入
        #   第二个 y：Key 和 Value 的输入
        #
        # 因为 Query、Key、Value 都来自同一个 Token 序列，
        # 所以这里执行的是 Self-Attention（自注意力）
        #
        # 输入：
        #   Query 输入：[batch_size, seq_len, features]
        #   Key/Value 输入：[batch_size, seq_len, features]
        #
        # 假设：
        #   features = 768
        #   num_heads = 12
        #   head_dim = 768 // 12 = 64
        #
        # Q、K、V 线性投影后的典型 Shape：
        #   Q：[batch_size, seq_len, num_heads, head_dim]
        #   K：[batch_size, seq_len, num_heads, head_dim]
        #   V：[batch_size, seq_len, num_heads, head_dim]
        #
        # Attention 权重：
        #   [batch_size, num_heads, seq_len, seq_len]
        #
        # 多头注意力输出：
        #   [batch_size, seq_len, features]
        #
        # out["sa"]：保存 Self-Attention 的输出
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)

        # 对 Self-Attention 输出施加设备分片约束
        #
        # 输入 y：[batch_size, seq_len, features]
        # 输出 y：[batch_size, seq_len, features]
        #
        # 不改变逻辑 Shape
        y = sharding.activation_sharding_constraint(y)

        # 对 Self-Attention 输出进行 Dropout
        #
        # rate=self.dropout：随机丢弃概率
        # deterministic：是否关闭 Dropout
        #
        # 输入 y：[batch_size, seq_len, features]
        # 输出 y：[batch_size, seq_len, features]
        #
        # Dropout 不改变 Shape
        y = nn.Dropout(rate=self.dropout)(y, deterministic)

        # 第一次残差连接（Residual Connection）
        #
        # x：进入 Self-Attention 分支之前的原始输入
        # y：经过 LayerNorm、Self-Attention、Dropout 后的输出
        #
        # x：[batch_size, seq_len, features]
        # y：[batch_size, seq_len, features]
        #
        # x + y：逐元素相加（Element-wise Addition）
        #
        # 两个张量 Shape 相同，不需要广播
        #
        # 计算公式：
        #   x_new = x_old + Dropout(SelfAttention(LayerNorm(x_old)))
        #
        # 输出：
        #   x：[batch_size, seq_len, features]
        #
        # out["+sa"]：保存第一次残差连接后的结果
        x = out["+sa"] = x + y

        # ============================================================
        # 2. MLP（多层感知机）
        # ============================================================

        # 对第一次残差连接后的 x 再次进行 LayerNorm
        #
        # 输入 x：[batch_size, seq_len, features]
        # 输出 y：[batch_size, seq_len, features]
        #
        # 注意：
        #   这里使用的是新的 LayerNorm 子模块，
        #   与 Self-Attention 前面的 LayerNorm 参数不共享
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)

        # MlpBlock：多层感知机模块
        #
        # mlp_dim=self.mlp_dim：
        #   MLP 中间隐藏层的特征维度
        #
        # dropout=self.dropout：
        #   MLP 内部的 Dropout 概率
        #
        # dtype_mm=self.dtype_mm：
        #   MLP 内部矩阵运算使用的数据类型
        #
        # (y, deterministic)：
        #   y：输入 Token 特征
        #   deterministic：是否关闭 Dropout
        #
        # 输入 y：[batch_size, seq_len, features]
        #
        # 假设：
        #   features = 768
        #   mlp_dim = 3072
        #
        # MLP 内部典型 Shape 变化：
        #   [batch_size, seq_len, 768]
        #       ↓
        #   [batch_size, seq_len, 3072]
        #       ↓
        #   [batch_size, seq_len, 768]
        #
        # 输出 y：[batch_size, seq_len, features]
        #
        # out["mlp"]：保存 MLP 的输出
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)

        # 对 MLP 输出施加设备分片约束
        #
        # 输入 y：[batch_size, seq_len, features]
        # 输出 y：[batch_size, seq_len, features]
        y = sharding.activation_sharding_constraint(y)

        # 对 MLP 输出再次进行 Dropout
        #
        # 输入 y：[batch_size, seq_len, features]
        # 输出 y：[batch_size, seq_len, features]
        y = nn.Dropout(rate=self.dropout)(y, deterministic)

        # 第二次残差连接（Residual Connection）
        #
        # x：第一次残差连接后的特征
        # y：经过第二次 LayerNorm、MLP、Dropout 后的输出
        #
        # x：[batch_size, seq_len, features]
        # y：[batch_size, seq_len, features]
        #
        # x + y：逐元素相加，不需要广播
        #
        # 计算公式：
        #   x_new = x_old + Dropout(MLP(LayerNorm(x_old)))
        #
        # 输出 x：[batch_size, seq_len, features]
        #
        # out["+mlp"]：保存第二次残差连接后的结果
        x = out["+mlp"] = x + y

        # 对 Transformer Block 最终输出施加设备分片约束
        #
        # x：[batch_size, seq_len, features]
        #
        # 不改变逻辑 Shape
        x = sharding.activation_sharding_constraint(x)

        # 返回：
        #
        # x：Transformer Encoder Block 的最终输出
        #   [batch_size, seq_len, features]
        #
        # out：保存中间计算结果的字典，每个 value值：[batch_size, seq_len, features]
        #   out["sa"]：Self-Attention 输出
        #   out["+sa"]：第一次残差连接后的输出
        #   out["mlp"]：MLP 输出
        #   out["+mlp"]：第二次残差连接后的输出
        return x, out


class Encoder(nn.Module):
    """Transformer Encoder：由多个 Encoder1DBlock 堆叠组成的编码器"""

    # depth：Transformer Encoder Block 的层数
    # 例如：depth=12，表示依次执行 12 个 Encoder1DBlock
    # 每层包含 Multi-Head Self-Attention、MLP 和残差连接
    depth: int

    # mlp_dim：MLP（Multi-Layer Perceptron）的隐藏层维度
    # None：默认使用输入特征维度 features 的 4 倍
    # 例如：features=768 → mlp_dim=3072
    # Shape：[batch_size, seq_len, 768] → [batch_size, seq_len, 3072] → [batch_size, seq_len, 768]
    mlp_dim: int | None = None

    # num_heads：多头注意力（Multi-Head Attention）的注意力头数量
    # 例如：features=768，num_heads=12 → head_dim=64
    # Q/K/V：[batch_size, seq_len, num_heads, head_dim]
    # 例如：[2, 197, 12, 64]
    num_heads: int = 12

    # dropout：Dropout 概率
    # 0.1：训练时随机丢弃约 10% 的元素
    # 0.0：不进行随机丢弃
    # Dropout 不改变 Shape
    dropout: float = 0.0

    # scan：是否使用 JAX/Flax Scan 机制
    # True：通过 nn.scan 将多个 Encoder1DBlock 组织为循环结构
    # False：使用 Python for 循环逐层创建 Encoder1DBlock
    # Scan 可减少循环体在编译计算图中的重复展开
    # 不改变网络层数及最终输出 Shape
    scan: bool = False

    # remat_policy：重计算策略（Rematerialization Policy）
    # Rematerialization：反向传播时重新计算部分中间结果，以减少显存占用
    # "nothing_saveable"：不允许保存重计算区域内的中间残差
    # 通过计算时间换取显存空间
    # 本代码仅在 scan=True 时使用该策略
    remat_policy: str = "nothing_saveable"

    # dtype_mm：矩阵运算的数据类型（Matrix Multiplication Dtype）
    # "float32"：32 位浮点数
    # "float16"：16 位浮点数
    # "bfloat16"：Brain Floating Point 16
    # 控制 Encoder 内部相关模块的计算数据类型，不改变 Shape
    dtype_mm: str = "float32"

    # @nn.compact：Flax Linen 装饰器
    # 允许在 __call__() 中直接创建子模块和参数
    @nn.compact
    def __call__(
        # self：当前 Encoder 实例
        self,

        # x：输入 Token 特征
        # Shape：[batch_size, seq_len, features]
        # 例如：[2, 197, 768]
        x,

        # deterministic：是否使用确定性模式
        # True：关闭 Dropout，通常用于推理
        # False：允许启用 Dropout，通常用于训练
        deterministic=True,
    ):

        # out = {
        #     "block00": {
        #         "sa":   ...,  # Shape：[batch_size, seq_len, features]
        #         "+sa":  ...,  # Shape：[batch_size, seq_len, features]
        #         "mlp":  ...,  # Shape：[batch_size, seq_len, features]
        #         "+mlp": ...,  # Shape：[batch_size, seq_len, features] 
        #     },
        #     "block01": {
        #         "sa":   ...,  # Shape：[batch_size, seq_len, features]
        #         "+sa":  ...,  # Shape：[batch_size, seq_len, features]
        #         "mlp":  ...,  # Shape：[batch_size, seq_len, features]
        #         "+mlp": ...,  # Shape：[batch_size, seq_len, features] 
        #     },
        #     ...
        # }
        out = {}

        # ============================================================
        # 1. scan=True：使用 Scan + Rematerialization
        # ============================================================

        if self.scan:

            # nn.remat：Rematerialization（重计算）
            # 对 Encoder1DBlock 进行包装，使部分中间结果在反向传播时重新计算
            # 减少训练过程中需要保存的激活值，从而降低显存占用
            #
            # block：经过 nn.remat 包装后的 Encoder1DBlock 模块
            # 注意：此处只是构造模块变换，还没有执行前向传播
            block = nn.remat(
                # target：需要进行重计算包装的模块
                # Encoder1DBlock：单层 Transformer Encoder Block
                Encoder1DBlock,

                # prevent_cse：是否阻止公共子表达式消除
                # CSE = Common Subexpression Elimination（公共子表达式消除）
                # False：不阻止编译器对相同计算进行公共子表达式消除
                # True：尝试阻止相关优化
                prevent_cse=False,

                # static_argnums：指定 __call__() 中的静态参数索引
                # 静态参数：在 JAX 跟踪、编译时作为静态值处理
                #
                # Encoder1DBlock.__call__(self, x, deterministic=True)
                # 索引：
                #   0 → self
                #   1 → x
                #   2 → deterministic
                #
                # (2,)：将 deterministic 作为静态参数
                # 注意：(2,) 是单元素元组，不能写成 (2)
                static_argnums=(2,),

                # policy：Checkpoint / Rematerialization 的保存策略
                #
                # jax.checkpoint_policies：JAX 提供的重计算策略集合
                # self.remat_policy：策略名称字符串
                #
                # getattr(object, name, default)：
                #   获取 object 中名称为 name 的属性
                #   属性不存在时返回 default
                #
                # 例如：
                # self.remat_policy="nothing_saveable"
                # → jax.checkpoint_policies.nothing_saveable
                #
                # None：策略名称不存在时使用的默认值
                policy=getattr(
                    # object：保存重计算策略的对象
                    jax.checkpoint_policies,

                    # name：需要查找的策略名称
                    self.remat_policy,

                    # default：属性不存在时返回 None
                    None,
                ),
            )

            # ========================================================
            # 2. nn.scan：循环执行 depth 个 Encoder1DBlock
            # ========================================================

            # nn.scan 规定返回两个部分，因此 nn.scan 用于迭代的模块也要具有两个返回值
            # 返回值：
            #   1. final_carry：保留最后一层迭代的 carry 输出
            #   2. stacked_outputs：收集每一层的输出
            #
            # 这里的输出：
            #   1. x（final_carry）：最后一层迭代的 carry 输出
            #      x：[batch_size, seq_len, features]
            #   2. scan_out（stacked_outputs）：收集每一层的输出
            #      每一层的输出 out = {
            #          "sa":   [batch_size, seq_len, features],
            #          "+sa":  [batch_size, seq_len, features],
            #          "mlp":  [batch_size, seq_len, features],
            #          "+mlp": [batch_size, seq_len, features],
            #      } 
            #   
            #      scan_out = {
            #          "sa":   [depth, batch_size, seq_len, features],
            #          "+sa":  [depth, batch_size, seq_len, features],
            #          "mlp":  [depth, batch_size, seq_len, features],
            #          "+mlp": [depth, batch_size, seq_len, features],
            #      } 
            x, scan_out = nn.scan(

                ################################ 
                # 1. 配置 nn.scan 的循环规则
                ################################ 
                
                # target：指定每次 Scan 迭代执行的模块
                # block：经过 nn.remat 包装的 Encoder1DBlock
                # 每次执行相同的模块结构，但可以使用不同的模型参数
                target = block,

                # variable_axes：指定 Flax 变量集合沿哪个轴组织
                # "params"：Flax 中存放可训练参数的变量集合
                # 0：将参数张量的第 0 维作为 Scan 的循环维度
                #
                # 例如：
                # 单层 Dense 权重：[768, 3072]
                # depth=12 时：[12, 768, 3072]
                #
                # 第 1 层使用 params[0]
                # 第 2 层使用 params[1]
                # ...
                # 第 12 层使用 params[11]
                #
                # 每层使用独立的参数，不共享权重
                variable_axes={"params": 0},

                # split_rngs：控制是否沿 Scan 迭代为每层拆分独立随机密钥
                # RNG = Random Number Generator（随机数生成器）
                #
                # 1. "params"：控制模型参数初始化使用的随机密钥
                #    1) True：每层使用不同的参数初始化随机密钥
                #    2) False：每层使用相同的参数初始化随机密钥
                #
                # 2. "dropout"：控制 Dropout 掩码采样使用的随机密钥
                #    1) True：每层使用不同的 Dropout 随机密钥
                #    2) False：每层使用相同的 Dropout 随机密钥
                #
                # 注意：
                # split_rngs 控制随机密钥是否按层拆分，不控制权重是否共享
                split_rngs={
                    "params": True,
                    "dropout": True,
                },

                # in_axes：指定非 carry 输入如何沿 Scan 迭代传递
                #
                # block.__call__(self, x, deterministic)
                #
                # 1. self：模块实例，不属于 __call__ 的扫描数据输入
                #
                # 2. x：carry（循环传递状态）
                #    默认将第一个非 self 位置参数作为 carry
                #    第 1 层：输入 x0 → 输出 x1
                #    第 2 层：输入 x1 → 输出 x2
                #    第 3 层：输入 x2 → 输出 x3
                #
                # 3. deterministic：非 carry 输入
                #    True：关闭 Dropout
                #    False：训练时允许启用 Dropout
                #
                # nn.broadcast：所有迭代接收相同的非 carry 输入
                # 例如 deterministic=False：
                # 第 1 层：False
                # 第 2 层：False
                # 第 3 层：False
                #
                # 注意：in_axes 不控制 carry，carry 自动逐层传递
                in_axes=nn.broadcast,

                # 将每次迭代返回的 outputs 沿第 0 维堆叠，形成 stacked_outputs
                out_axes = 0,

                # length：Scan 循环执行的次数
                # self.depth：Transformer Encoder Block 的层数
                # 例如：depth=12 → 执行 12 次 Encoder1DBlock
                length=self.depth,
            )(
                ############################################################### 
                # 2. 创建 Scan 模块实例，并设置所有 Encoder Block 共同使用的模块配置
                ############################################################### 
                
                # name：Scan 模块的名称
                # 用于 Flax 参数树中的模块命名
                name="encoderblock",

                # dtype_mm：各层矩阵运算的数据类型
                dtype_mm=self.dtype_mm,

                # mlp_dim：各层 MLP 的隐藏层维度
                mlp_dim=self.mlp_dim,

                # num_heads：各层 Attention Head 的数量
                num_heads=self.num_heads,

                # dropout：各层 Dropout 概率
                dropout=self.dropout,
            )(
                ############################################################### 
                # 3. 执行前向传播，传入 x 和 deterministic
                ############################################################### 

                # x：第一层 Encoder1DBlock 的输入
                # Shape：[batch_size, seq_len, features]
                x,

                # deterministic：是否关闭 Dropout
                # 各层使用相同的确定性模式
                deterministic,
            )

            # ========================================================
            # 3. 将 scan_out 拆分为各层独立的中间结果
            # ========================================================

            # range(self.depth)：返回一个可迭代的 range 实例，0 ~ self.depth-1
            # lyr：layer_index 表示每一层的下标
            for lyr in range(self.depth):

                # jax.tree.map() 是 JAX 提供的 PyTree 映射函数，它会遍历 PyTree 中的所有叶子节点，对每个叶子节点执行 f，不会修改原来的 PyTree，而是返回一个新的 PyTree，新 PyTree 保留原来的容器结构，但叶子节点的值可以改变
                # 基本语法：
                #   jax.tree.map(
                #       f,       # 对每个叶子节点执行的函数
                #       tree,    # 需要遍历的 PyTree
                #   )
                #
                # jax.tree.map() = {
                #     "sa":   ...,  # Shape：[depth, batch_size, seq_len, features]
                #     "+sa":  ...,  # Shape：[depth, batch_size, seq_len, features]
                #     "mlp":  ...,  # Shape：[depth, batch_size, seq_len, features]
                #     "+mlp": ...,  # Shape：[depth, batch_size, seq_len, features]
                # } 
                #
                # 02d：格式说明符
                #   1) 0：不足指定宽度时，在左侧补零
                #   2) 2：最小宽度为 2 位
                #   3) d：十进制整数（Decimal Integer）
                #
                # out = {
                #     "block00": {
                #         "sa":   ...,  # Shape：[batch_size, seq_len, features]
                #         "+sa":  ...,  # Shape：[batch_size, seq_len, features]
                #         "mlp":  ...,  # Shape：[batch_size, seq_len, features]
                #         "+mlp": ...,  # Shape：[batch_size, seq_len, features] 
                #     },
                #     "block01": {
                #         "sa":   ...,  # Shape：[ batch_size, seq_len, features]
                #         "+sa":  ...,  # Shape：[ batch_size, seq_len, features]
                #         "mlp":  ...,  # Shape：[ batch_size, seq_len, features]
                #         "+mlp": ...,  # Shape：[ batch_size, seq_len, features] 
                #     },
                #     ...
                # }
                out[f"block{lyr:02d}"] = jax.tree.map(

                    # 对每个叶子节点执行的函数
                    # 此时的 o 就是 scan_out 中每一组键值对的值
                    # o：[depth, batch_size, seq_len, features]
                    # 由于新 PyTree 会保留原来的容器结构，所以键会自动保留
                    #
                    # lambda：取出每个叶子节点 o，然后取出叶子节点的第 lyr 层结果
                    lambda o, lyr=lyr: o[lyr],

                    # 需要遍历的 PyTree
                    # scan_out = {
                    #     "sa":   ...,  # Shape：[depth, batch_size, seq_len, features]
                    #     "+sa":  ...,  # Shape：[depth, batch_size, seq_len, features]
                    #     "mlp":  ...,  # Shape：[depth, batch_size, seq_len, features]
                    #     "+mlp": ...,  # Shape：[depth, batch_size, seq_len, features] 
                    # } 
                    scan_out,
                )

        # ============================================================
        # 4. scan = False：使用普通 for 循环逐层构建 Encoder
        # ============================================================

        else:
            # 使用 for 循环，创建一层 Encoder，使用一层 Encoder

            # range(self.depth)：返回一个可迭代的 range 实例，0 ~ self.depth-1
            # lyr：layer_index 表示每一层的下标
            for lyr in range(self.depth):

                block_cur = Encoder1DBlock(

                    # name：当前 Encoder Block 的模块名称
                    name=f"encoderblock_{lyr}",

                    # dtype_mm：当前 Block 的矩阵运算数据类型
                    dtype_mm=self.dtype_mm,

                    # mlp_dim：当前 Block 的 MLP 隐藏层维度
                    # None 时使用 4*features
                    mlp_dim=self.mlp_dim,

                    # num_heads：当前 Block 的 Attention Head 数量
                    num_heads=self.num_heads,

                    # dropout：当前 Block 的 Dropout 概率
                    dropout=self.dropout,
                )

                x, out[f"block{lyr:02d}"] = block_cur(
                    # x：当前层输入
                    # x：[batch_size, seq_len, features]
                    x,

                    # deterministic：是否使用 Dropout
                    deterministic,
                )

            # pre_ln：Pre-LayerNorm Output（最终归一化前的输出）
            #
            # 保存最后一层 Encoder1DBlock 的输出
            # 此时尚未执行 Encoder 最后的 LayerNorm
            #
            # x：[batch_size, seq_len, features]
            out["pre_ln"] = x

        # ============================================================
        # 5. 最终 LayerNorm + 返回结果
        # ============================================================

        # nn.LayerNorm：Layer Normalization（层归一化）
        #
        # 对所有 Encoder Block 的最终输出进行归一化
        # 默认沿最后一个特征维度 features 计算均值和方差
        #
        # 输入 x：[batch_size, seq_len, features]
        # 输出 x：[batch_size, seq_len, features]
        #
        # 注意：
        # 这里的 LayerNorm 是整个 Encoder 的最终归一化层
        # 与各个 Encoder1DBlock 内部的 LayerNorm 参数不共享
        #
        # 返回值：
        # x：[batch_size, seq_len, features]
        # out = {
        #     "block00": {
        #         "sa":   ...,  # Shape：[batch_size, seq_len, features]
        #         "+sa":  ...,  # Shape：[batch_size, seq_len, features]
        #         "mlp":  ...,  # Shape：[batch_size, seq_len, features]
        #         "+mlp": ...,  # Shape：[batch_size, seq_len, features] 
        #     },
        #     "block01": {
        #         "sa":   ...,  # Shape：[batch_size, seq_len, features]
        #         "+sa":  ...,  # Shape：[batch_size, seq_len, features]
        #         "mlp":  ...,  # Shape：[batch_size, seq_len, features]
        #         "+mlp": ...,  # Shape：[batch_size, seq_len, features] 
        #     },
        #     ...
        # }
        return nn.LayerNorm(
            # name：最终 LayerNorm 子模块的名称
            name="encoder_norm",

            # dtype：归一化模块的计算及输出数据类型
            dtype=self.dtype_mm,
        )(
            # x：最后一个 Encoder1DBlock 的输出
            # x：[batch_size, seq_len, features]
            x,
        ), out

# MAPHead：Multihead Attention Pooling（多头注意力池化）
# 通过可学习的 Probe Token，使用 Cross-Attention 聚合所有 Patch Tokens 的特征
class MAPHead(nn.Module):

    # MLP 中间隐藏层维度，None 表示默认使用输入 features 的 4 倍
    mlp_dim: int | None = None

    # Attention Head 数量
    num_heads: int = 12

    # 矩阵运算的数据类型
    dtype_mm: str = "float32"

    # 允许在 __call__() 中创建参数和子模块
    @nn.compact
    def __call__(
        self,
        x,
    ):
        # 参数：
        #   self：当前 MAPHead 模块实例
        #   x：Encoder 输出的 Patch Tokens，[batch_size, seq_len, features]

        # n = batch_size，_ = seq_len（不使用），d = features
        n, _, d = x.shape

        # self.param()：创建或读取名为 "probe" 的可学习参数
        # Xavier Uniform：初始化方法；x.dtype：参数数据类型
        # probe：[1, 1, features]，作为 Cross-Attention 的 Query 输入
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)

        # jnp.tile(probe, [n, 1, 1])：沿 batch_size 维度复制 n 次，其余维度不变
        # probe：[1, 1, features] → [batch_size, 1, features]
        # 所有样本共享同一组可学习 Probe 参数，但 Attention 输出因图像而异
        probe = jnp.tile(probe, [n, 1, 1])

        # Multi-Head Dot-Product Attention（多头点积注意力）
        # num_heads：Attention Head 数量
        # dtype：Attention 计算的数据类型
        # kernel_init：使用 Xavier Uniform 初始化注意力投影权重
        #
        # (probe, x)：Query 来自 probe，Key/Value 来自 x，因此是 Cross-Attention
        # Query 输入：[batch_size, 1, features]
        # Key/Value 输入：[batch_size, seq_len, features]
        #
        # 假设 features=768，num_heads=12，head_dim=768//12=64：
        # Q：[batch_size, 1, 12, 64]
        # K：[batch_size, seq_len, 12, 64]
        # V：[batch_size, seq_len, 12, 64]
        # Attention 权重：[batch_size, 12, 1, seq_len]
        #
        # 作用：Probe 对所有 Patch Tokens 计算注意力权重，
        #      根据权重对 Value 加权求和，将所有 Token 信息聚合到一个 Token 中
        # x：[batch_size, 1, features]
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        # LayerNorm：沿 features 维度归一化
        # x：[batch_size, 1, features] → y：[batch_size, 1, features]
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)

        # MLP + Residual Connection（残差连接）
        # MlpBlock：对归一化后的特征进行非线性变换，输出维度不变
        # x = Attention 输出 + MLP 输出
        # x：[batch_size, 1, features]
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype_mm=self.dtype_mm)(y)

        # 提取唯一的全局 Token，作为整张图像的全局特征向量
        # x[:, 0]：[batch_size, 1, features] → [batch_size, features]
        return x[:, 0]

    # Eg：
    # {**decode_variant(variant), **kw} = {
    #     "patch_size": (14, 14),  # 每个图像 Patch 的大小
    #     "width": 1152,           # Transformer 隐藏层维度
    #     "depth": 27,             # Transformer Block 层数
    #     "mlp_dim": 4304,         # MLP 中间隐藏层维度
    #     "num_heads": 16,         # Attention Head 数量
    #     "pool_type": "none",     # 不进行池化，保留所有 Patch Token
    #     "scan": True,            # 使用 nn.scan 构建 Transformer Encoder
    #     "dtype_mm": "bfloat16",  # 矩阵运算的数据类型
    # }
    # 将 {**decode_variant(variant), **kw} 用 ** 解包作为参数传入
    # return _Module(num_classes, **{**decode_variant(variant), **kw})

# Vision Transformer 模型
class _Module(nn.Module):

    # num_classes：分类类别数量
    # 1. int：输出类别数量，例如 1000
    # 2. None：不创建最终的分类 Head，保留特征表示
    num_classes: int | None = None

    # patch_size：每个图像 Patch（图像块）的大小
    # 1. Sequence[int]：包含 Patch 高度和宽度的序列
    # 2. 默认 (16, 16)，表示每个 Patch 的大小为 16×16 像素
    patch_size: Sequence[int] = (16, 16)

    # width：Transformer 的隐藏特征维度，也是 Patch Embedding 的输出通道数量
    # 例如 width=768，则每个 Patch Token 对应 768 维特征
    width: int = 768

    # depth：Transformer Encoder 中的 Block 数量
    # 例如 depth=12，表示包含 12 层 Transformer Block
    depth: int = 12

    # mlp_dim：Transformer Block 中 FeedForward / MLP 的中间隐藏维度
    # None：由 Encoder 内部决定默认值，通常为 width 的 4 倍，例如 width=768，则常见的 mlp_dim=3072
    mlp_dim: int | None = None

    # num_heads：Multi-Head Attention（多头注意力）的 Head 数量
    # 例如 num_heads=12，当 width=768 且平均划分时，每个 Head 的维度为 768/12=64
    num_heads: int = 12

    # posemb：Position Embedding（位置嵌入）的类型
    # 1. "learn"：使用可学习的位置嵌入
    # 2. "sincos2d"：使用二维正弦余弦位置编码
    posemb: str = "learn"

    # rep_size：Representation Size（表示层的输出特征维度）
    # 1. int：指定表示层的输出维度，例如 rep_size=512，输出维度为 512
    #    非零整数在 if 判断中视为 True，因此会创建表示层
    # 2. bool：
    #    1) True：创建输出维度为 width 的表示层
    #    2) False（默认）：不创建额外的表示层
    rep_size: int | bool = False

    # dropout：Dropout Rate（随机丢弃概率）
    # 1. 0.0：不进行随机丢弃
    # 2. 0.1：训练时以 10% 的概率丢弃部分特征
    dropout: float = 0.0

    # pool_type：Pooling Type（特征池化方式）
    # 1. "gap"：Global Average Pooling，对所有 Token 取平均
    # 2. "map"：Multihead Attention Pooling，使用注意力池化
    # 3. "tok"：使用额外添加的 CLS Token 进行池化
    # 4. "0"：直接使用第一个 Token 的隐藏特征
    # 5. "none"：不执行池化，保留所有 Token
    pool_type: str = "gap"

    # head_zeroinit：是否将分类 Head 的权重初始化为全零
    # 1. True：分类 Head 的 kernel 使用全零初始化
    # 2. False：使用 nn.Dense 默认的 kernel 初始化方式
    head_zeroinit: bool = True

    # scan：是否在 Encoder 内部使用 nn.scan() 组织 Transformer Block
    # 1. True：使用 scan 形式
    # 2. False：使用非 scan 形式
    # 具体实现由 Encoder 决定
    scan: bool = False

    # remat_policy：Gradient Checkpointing（梯度检查点）的重计算策略
    # 1. "nothing_saveable"：不保存被策略控制的中间结果，反向传播时重新计算
    # 2. "dots_with_no_batch_dims_saveable"：允许保存部分 dot 运算结果，通常用更多显存换取更少的重计算
    remat_policy: str = "nothing_saveable"

    # dtype_mm：Matrix Multiplication Data Type（矩阵乘法计算数据类型）
    # "float32"：使用 32 位浮点数
    # 也可以配置为 "bfloat16" 等较低精度类型
    # 注意：不代表所有模型参数都使用该数据类型
    dtype_mm: str = "float32"

    # @nn.compact 允许在 __call__() 方法内部创建参数或子模块
    # 例如 self.param() 和 nn.Dense() 都可以在该方法中使用
    @nn.compact
    def __call__(
        self, 
        # 图像输入：[batch_size, height, width, channels]
        image, 
        *, 
        # train：是否处于训练模式
        # True：训练模式，允许执行 Dropout
        # False：推理模式，关闭 Dropout
        train=False
    ):
        # out：保存 ViT 各阶段中间结果的字典
        out = {}

        # 将 image 转成 float32 格式
        image = jnp.asarray(image, jnp.float32)

        # kernal：[patch_height, patch_width, channels, features]
        # x：[batch_size, patch_rows, patch_cols, features]
        #
        # batch_size：图像个数
        # patch_height：单个 Patch 的高度
        # patch_width：单个 Patch 的宽度
        # patch_rows：Patch 的行数
        # patch_cols：Patch 的列数
        # channels：输入图像的通道数
        # features：每个 Patch 输出的特征维度
        #
        # out：保存 ViT 各阶段中间结果的字典
        # 
        # out = {
        #     # 1. Patch Embedding：卷积提取的 Patch 特征
        #     "stem": ...,  # Shape：[batch_size, patch_rows, patch_cols, features]
        #
        #     # 2. Patch Tokens + Positional Embedding（此时尚未添加 CLS Token）
        #     "with_posemb": ...,  # Shape：[batch_size, patch_rows * patch_cols, features]
        # } 
        x = out["stem"] = nn.Conv(
            # 卷积核个数
            features = self.width,
            # 卷积核的大小
            kernel_size = self.patch_size,
            # 步长
            strides=self.patch_size,
            # 不进行边界填充
            padding="VALID", 
            # 卷积子模块名称
            name="embedding",
            # 卷积计算使用的数据类型
            dtype=jnp.float32,
        )(image)

        # n = batch_size
        # h = patch_rows
        # w = patch_cols
        # c = features
        n, h, w, c = x.shape
        # x：[batch_size, patch_rows * patch_cols, features]
        x = jnp.reshape(x, [n, h * w, c])

        # out：保存 ViT 各阶段中间结果的字典
        # 
        # out = {
        #     # 1. Patch Embedding：卷积提取的 Patch 特征
        #     "stem": ...,  # Shape：[batch_size, patch_rows, patch_cols, features]
        #
        #     # 2. Patch Tokens + Positional Embedding（此时尚未添加 CLS Token）
        #     "with_posemb": ...,  # Shape：[batch_size, patch_rows * patch_cols, features]
        # } 
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)

        # 使用 CLS Token（Classification Token）作为整张图像的全局特征表示
        # 分为两个阶段：
        # Encoder 前：添加 CLS Token
        #   1) 创建一个可学习的 CLS Token，Shape 为 [1, 1, features]
        #   2) 将其复制到每个样本，并拼接到所有 Patch Tokens 的最前面
        #   3) x 的 Shape 从 [batch_size, patch_rows * patch_cols, features] 变为 
        #      [batch_size, 1 + patch_rows * patch_cols, features]
        # 
        # Encoder 后：提取 CLS Token
        #   1) CLS Token 经过 Transformer 的 Self-Attention，与 Patch Tokens 交互，学习图像的全局特征
        #   2) 使用 x[:, 0] 提取 CLS Token，Shape 为 [batch_size, features]
        #   3) 使用 encoded[:, 1:] 去掉 CLS Token，保留所有 Patch Tokens 的特征
        if self.pool_type == "tok":

            # cls：[1, 1, features]
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)

            # jnp.tile(cls, [n, 1, 1])：将 cls 沿着 batch_size 维度复制 n 次
            # n：沿着 batch_size 维度复制 n 次
            # 1：Token 维度不复制
            # 1：features 维度不复制
            # 输入：
            #   1) jnp.tile(cls, [n, 1, 1])：[batch_size, 1, features]
            #   2) x：[batch_size, patch_rows * patch_cols, features]
            # 输出（不会发生广播）：
            #   x：[batch_size, 1 + patch_rows * patch_cols, features]
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        # n = batch_size, 
        # c = features
        n, _, c = x.shape 

        # 随机丢弃
        x = nn.Dropout(rate=self.dropout)(x, not train)

        # 使用较低精度的数据类型可以减少显存占用，
        # 并可能提高后续 Transformer 矩阵运算的速度。
        x = x.astype(self.dtype_mm)

        # out：保存 ViT 各阶段中间结果的字典
        #
        # seq_len：
        #   pool_type == "tok"：1 + patch_rows * patch_cols
        #   pool_type != "tok"：patch_rows * patch_cols
        #
        # out = {
        #     # 1. Patch Embedding：卷积提取的 Patch 特征
        #     "stem": ...,  # Shape：[batch_size, patch_rows, patch_cols, features]
        #
        #     # 2. Patch Tokens + Positional Embedding（此时尚未添加 CLS Token）
        #     "with_posemb": ...,  # Shape：[batch_size, patch_rows * patch_cols, features]
        #
        #     # 3. 各层 Encoder Block 的中间结果
        #     "encoder": {
        #         "block00": {
        #             "sa":   ...,  # Shape：[batch_size, seq_len, features]
        #             "+sa":  ...,  # Shape：[batch_size, seq_len, features]
        #             "mlp":  ...,  # Shape：[batch_size, seq_len, features]
        #             "+mlp": ...,  # Shape：[batch_size, seq_len, features]
        #         },
        #         "block01": {
        #             "sa":   ...,  # Shape：[batch_size, seq_len, features]
        #             "+sa":  ...,  # Shape：[batch_size, seq_len, features]
        #             "mlp":  ...,  # Shape：[batch_size, seq_len, features]
        #             "+mlp": ...,  # Shape：[batch_size, seq_len, features]
        #         },
        #         # ...
        #     },
        # } 
        x, out["encoder"] = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            scan=self.scan,
            remat_policy=self.remat_policy,
            dtype_mm=self.dtype_mm,
            name="Transformer",
        )(x, deterministic=not train)

        # out：保存 ViT 各阶段中间结果的字典
        #
        # seq_len：
        #   pool_type == "tok"：1 + patch_rows * patch_cols
        #   pool_type != "tok"：patch_rows * patch_cols
        #
        # output_features：
        #   self.rep_size 为 False：features
        #   self.rep_size 为 True：features（rep_size = self.width）
        #   self.rep_size 为正整数：rep_size
        #
        # out = {
        #     # 1. Patch Embedding：卷积提取的 Patch 特征
        #     "stem": ...,  # Shape：[batch_size, patch_rows, patch_cols, features]
        #
        #     # 2. Patch Tokens + Positional Embedding（此时尚未添加 CLS Token）
        #     "with_posemb": ...,  # Shape：[batch_size, patch_rows * patch_cols, features]
        #
        #     # 3. 各层 Encoder Block 的中间结果
        #     "encoder": {
        #         "block00": {
        #             "sa":   ...,  # Shape：[batch_size, seq_len, features]
        #             "+sa":  ...,  # Shape：[batch_size, seq_len, features]
        #             "mlp":  ...,  # Shape：[batch_size, seq_len, features]
        #             "+mlp": ...,  # Shape：[batch_size, seq_len, features]
        #         },
        #         "block01": {
        #             "sa":   ...,  # Shape：[batch_size, seq_len, features]
        #             "+sa":  ...,  # Shape：[batch_size, seq_len, features]
        #             "mlp":  ...,  # Shape：[batch_size, seq_len, features]
        #             "+mlp": ...,  # Shape：[batch_size, seq_len, features]
        #         },
        #         # ...
        #     },
        #
        #     # 4. 所有 Encoder Block + 最终 LayerNorm 的输出
        #     # pool_type == "tok" 时包含 CLS Token
        #     "encoded": ...,  # Shape：[batch_size, seq_len, features]
        # }
        encoded = out["encoded"] = x

        # 根据 pool_type 选择不同的池化方式，将 Encoder 输出的多个 Token 特征
        # 聚合或选取为一个代表整张图像的全局特征向量（pool_type="none" 除外）
        #
        # 输入 x：经过所有 Encoder Block + 最终 LayerNorm 的输出
        # Shape：[batch_size, seq_len, features]
        #
        # seq_len：
        #   pool_type == "tok"：1 + patch_rows * patch_cols
        #   其他情况：patch_rows * patch_cols
        #
        # 池化方式：
        #   "map"：通过 Cross-Attention 聚合所有 Patch Tokens 的特征
        #   "gap"：对所有 Patch Tokens 的特征求平均
        #   "0"：提取第 0 个 Patch Token 的特征
        #   "tok"：提取通过 Self-Attention 聚合图像信息的 CLS Token
        #   "none"：不进行池化，保留所有 Patch Tokens 的特征
        #
        # 输出 x：
        #   pool_type != "none"：[batch_size, features]
        #   pool_type == "none"：[batch_size, seq_len, features]


        # 1. "map"：Multihead Attention Pooling（多头注意力池化）
        # 使用可学习的 Probe Token，通过 Cross-Attention 聚合所有 Patch Tokens 的特征
        if self.pool_type == "map":

            # MAPHead：
            #   num_heads：Attention Head 的数量
            #   mlp_dim：MLP 中间隐藏层的特征维度
            #   dtype_mm：矩阵运算使用的数据类型
            #
            # 输入 x：[batch_size, patch_rows * patch_cols, features]
            # 输出 x = out["head_input"]：[batch_size, features]
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype_mm=self.dtype_mm,
            )(x)


        # 2. "gap"：Global Average Pooling（全局平均池化）
        elif self.pool_type == "gap":

            # axis=1：沿 seq_len 维度，对每张图像的所有 Patch Tokens
            #         在每个 features 维度上分别求平均，得到全局特征向量
            #
            # 输入 x：[batch_size, patch_rows * patch_cols, features]
            # 输出 x = out["head_input"]：[batch_size, features]
            x = out["head_input"] = jnp.mean(x, axis=1)


        # 3. "0"：提取第 0 个 Patch Token 的特征，作为整张图像的全局特征
        elif self.pool_type == "0":

            # x[:, 0]：保留所有 batch_size，提取 seq_len 维度中索引为 0 的 Token
            #
            # 输入 x：[batch_size, patch_rows * patch_cols, features]
            # 输出 x = out["head_input"]：[batch_size, features]
            #
            # 注意：该 Token 经过多层 Self-Attention，已经与其他 Patch Tokens 交互信息
            x = out["head_input"] = x[:, 0]


        # 4. "tok"：提取 CLS Token 作为整张图像的全局特征
        elif self.pool_type == "tok":

            # CLS Token：经过多层 Encoder，通过 Self-Attention 聚合 Patch Tokens 的信息
            # x[:, 0]：提取每张图像的 CLS Token
            #
            # 输入 x：[batch_size, 1 + patch_rows * patch_cols, features]
            # 输出 x = out["head_input"]：[batch_size, features]
            x = out["head_input"] = x[:, 0]

            # encoded[:, 1:]：剔除 CLS Token，只保留经过 Encoder 的 Patch Tokens
            #
            # 输入 encoded：[batch_size, 1 + patch_rows * patch_cols, features]
            # 输出 encoded：[batch_size, patch_rows * patch_cols, features]
            #
            # 注意：只修改局部变量 encoded，不改变之前保存的 out["encoded"]
            encoded = encoded[:, 1:]


        # 5. "none"：不进行池化，保留所有 Patch Tokens 的特征
        elif self.pool_type == "none":

            # pass：不执行任何操作，x 保持不变
            # 输入/输出 x：[batch_size, patch_rows * patch_cols, features]
            # 此分支不会创建 out["head_input"]
            pass


        # 6. 不支持的池化方式
        else:

            # raise ValueError：抛出参数值错误异常，并显示当前 pool_type
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        # out：保存 ViT 各阶段中间结果的字典
        #
        # seq_len：
        #   pool_type == "tok"：1 + patch_rows * patch_cols
        #   pool_type != "tok"：patch_rows * patch_cols
        #
        # output_features：
        #   self.rep_size 为 False：features
        #   self.rep_size 为 True：features（rep_size = self.width）
        #   self.rep_size 为正整数：rep_size
        #
        # out = {
        #     # 1. Patch Embedding：卷积提取的 Patch 特征
        #     "stem": ...,  # Shape：[batch_size, patch_rows, patch_cols, features]
        #
        #     # 2. Patch Tokens + Positional Embedding（此时尚未添加 CLS Token）
        #     "with_posemb": ...,  # Shape：[batch_size, patch_rows * patch_cols, features]
        #
        #     # 3. 各层 Encoder Block 的中间结果
        #     "encoder": {
        #         "block00": {
        #             "sa":   ...,  # Shape：[batch_size, seq_len, features]
        #             "+sa":  ...,  # Shape：[batch_size, seq_len, features]
        #             "mlp":  ...,  # Shape：[batch_size, seq_len, features]
        #             "+mlp": ...,  # Shape：[batch_size, seq_len, features]
        #         },
        #         "block01": {
        #             "sa":   ...,  # Shape：[batch_size, seq_len, features]
        #             "+sa":  ...,  # Shape：[batch_size, seq_len, features]
        #             "mlp":  ...,  # Shape：[batch_size, seq_len, features]
        #             "+mlp": ...,  # Shape：[batch_size, seq_len, features]
        #         },
        #         # ...
        #     },
        #
        #     # 4. 所有 Encoder Block + 最终 LayerNorm 的输出
        #     # pool_type == "tok" 时包含 CLS Token
        #     "encoded": ...,  # Shape：[batch_size, seq_len, features]
        #
        #     # 5. 池化后的全局特征向量（仅 pool_type != "none" 时存在）
        #     # "map"：Multihead Attention Pooling
        #     # "gap"：Global Average Pooling
        #     # "0"：第 0 个 Patch Token
        #     # "tok"：CLS Token
        #     "head_input": ...,  # Shape：[batch_size, features]
        # }

        # jnp.reshape()：将一维 Patch Token 序列恢复为二维 Patch 特征图
        #
        # encoded：经过所有 Encoder Block + 最终 LayerNorm 的 Patch Token 特征
        # n = batch_size：图像数量
        # h = patch_rows：Patch 特征图的行数
        # w = patch_cols：Patch 特征图的列数
        # -1：由 JAX 自动推断最后一维，即 features
        #
        # 输入 encoded：[batch_size, patch_rows * patch_cols, features]
        # 输出 x_2d：[batch_size, patch_rows, patch_cols, features]
        # 例如：[2, 196, 768] → [2, 14, 14, 768]
        #
        # 作用：恢复 Patch Tokens 的二维空间排列，不改变特征值
        x_2d = jnp.reshape(encoded, [n, h, w, -1])


        # rep_size：Representation Size（表示层的输出特征维度）
        # 1. int：指定表示层输出维度，例如 rep_size=512
        #    非零整数在 if 判断中视为 True，因此会创建表示层
        # 2. bool：
        #    1) True：创建输出维度为 self.width 的表示层
        #    2) False（默认）：不创建额外的表示层
        if self.rep_size:

            # 确定表示层的输出特征维度
            # 1. self.rep_size is True：rep_size = self.width（即 features）
            # 2. self.rep_size 为非零 int：rep_size = self.rep_size
            #
            # 注意：is True 严格判断布尔对象 True，不会将整数 1 当作 True
            rep_size = self.width if self.rep_size is True else self.rep_size

            # nn.Dense()：创建全连接层（Fully Connected Layer）
            #
            # rep_size：全连接层的输出特征维度
            # dtype：矩阵运算使用的数据类型
            # name="pre_logits"：全连接层的名称
            #
            # 输入最后一维：features
            # 权重 kernel：[features, rep_size]
            # 偏置 bias：[rep_size]
            # 输出最后一维：rep_size
            #
            # hid：全连接层模块，x_2d 和 x 共用同一组权重与偏置
            hid = nn.Dense(
                rep_size,
                dtype=self.dtype_mm,
                name="pre_logits",
            )

            # nn.tanh()：双曲正切激活函数（Hyperbolic Tangent）
            # 对 Dense 输出的每个元素进行非线性变换，数值范围为 (-1, 1)
            # 不改变张量 Shape
            #
            # x_2d → Dense → tanh
            # 输入 x_2d：[batch_size, patch_rows, patch_cols, features]
            # 输出 x_2d：[batch_size, patch_rows, patch_cols, rep_size]
            #
            # 作用：对每个 Patch 的特征向量进行相同的特征映射
            x_2d = nn.tanh(hid(x_2d))

            # x → Dense → tanh
            #
            # 输入 x：
            #   pool_type != "none"：[batch_size, features]
            #   pool_type == "none"：[batch_size, seq_len, features]
            #
            # 输出 x：
            #   pool_type != "none"：[batch_size, rep_size]
            #   pool_type == "none"：[batch_size, seq_len, rep_size]
            #
            # 作用：将最后一维 features 映射为 rep_size
            # 注意：与 x_2d 使用同一个 hid，因此共享权重和偏置
            x = nn.tanh(hid(x))


        # out：保存 ViT 各阶段中间结果的字典
        #
        # seq_len：
        #   pool_type == "tok"：1 + patch_rows * patch_cols
        #   pool_type != "tok"：patch_rows * patch_cols
        #
        # output_features：
        #   self.rep_size 为 False：features
        #   self.rep_size 为 True：features（rep_size = self.width）
        #   self.rep_size 为正整数：rep_size
        #
        # out = {
        #     # 1. Patch Embedding：卷积提取的 Patch 特征
        #     "stem": ...,  # Shape：[batch_size, patch_rows, patch_cols, features]
        #
        #     # 2. Patch Tokens + Positional Embedding（此时尚未添加 CLS Token）
        #     "with_posemb": ...,  # Shape：[batch_size, patch_rows * patch_cols, features]
        #
        #     # 3. 各层 Encoder Block 的中间结果
        #     "encoder": {
        #         "block00": {
        #             "sa":   ...,  # Shape：[batch_size, seq_len, features]
        #             "+sa":  ...,  # Shape：[batch_size, seq_len, features]
        #             "mlp":  ...,  # Shape：[batch_size, seq_len, features]
        #             "+mlp": ...,  # Shape：[batch_size, seq_len, features]
        #         },
        #         "block01": {
        #             "sa":   ...,  # Shape：[batch_size, seq_len, features]
        #             "+sa":  ...,  # Shape：[batch_size, seq_len, features]
        #             "mlp":  ...,  # Shape：[batch_size, seq_len, features]
        #             "+mlp": ...,  # Shape：[batch_size, seq_len, features]
        #         },
        #         # ...
        #     },
        #
        #     # 4. 所有 Encoder Block + 最终 LayerNorm 的输出
        #     # pool_type == "tok" 时包含 CLS Token
        #     "encoded": ...,  # Shape：[batch_size, seq_len, features]
        #
        #     # 5. 池化后的全局特征向量（仅 pool_type != "none" 时存在）
        #     # "map"：Multihead Attention Pooling
        #     # "gap"：Global Average Pooling
        #     # "0"：第 0 个 Patch Token
        #     # "tok"：CLS Token
        #     "head_input": ...,  # Shape：[batch_size, features]
        #
        #     # 6. Patch Tokens 的二维特征图（经过可选 Representation Layer）
        #     # 不进行池化，保留每个 Patch 的空间位置和特征
        #     "pre_logits_2d": ...,  # Shape：[batch_size, patch_rows, patch_cols, output_features]
        #
        #     # 7. 池化及可选 Representation Layer 处理后的特征
        #     # pool_type != "none"：[batch_size, output_features]
        #     # pool_type == "none"：[batch_size, seq_len, output_features]
        #     "pre_logits": ...,
        # }

        # 保存 Representation Layer 处理后的二维 Patch 特征图
        #
        # self.rep_size 为 False：
        #   out["pre_logits_2d"]：[batch_size, patch_rows, patch_cols, features]
        #
        # self.rep_size 为 True 或非零 int：
        #   out["pre_logits_2d"]：[batch_size, patch_rows, patch_cols, rep_size]
        out["pre_logits_2d"] = x_2d
        #
        # 保存池化及可选 Representation Layer 处理后的特征
        #
        # pool_type != "none"：
        #   self.rep_size 为 False：[batch_size, features]
        #   self.rep_size 为 True 或非零 int：[batch_size, rep_size]
        #
        # pool_type == "none"：
        #   self.rep_size 为 False：[batch_size, seq_len, features]
        #   self.rep_size 为 True 或非零 int：[batch_size, seq_len, rep_size]
        out["pre_logits"] = x

        # num_classes：分类类别数量（Number of Classes）
        # 将输入特征的最后一维（features 或 rep_size）映射为 num_classes，
        # 得到每个类别对应的原始分类分数（Logits）
        #
        # 1. self.rep_size 为 False：features → num_classes
        # 2. self.rep_size 为 True：features → num_classes
        # 3. self.rep_size 为正整数：rep_size → num_classes
        #
        # self.num_classes 为正整数时，创建分类层
        # self.num_classes 为 None 或 0 时，不创建分类层
        if self.num_classes:

            # head_zeroinit：是否将分类层（Classification Head）的权重初始化为全 0
            # 1. True：kw = {"kernel_init": nn.initializers.zeros}
            #    kernel_init：权重初始化函数，使用 zeros 将权重初始化为全 0
            # 2. False：kw = {}，使用 nn.Dense 默认的权重初始化方式
            #
            # kw：Keyword Arguments（关键字参数）字典
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}

            # nn.Dense()：创建分类层（Classification Head），将输入特征映射为分类分数（Logits）
            #
            # self.num_classes：分类类别数量，即输出特征维度
            # dtype=self.dtype_mm：矩阵运算使用的数据类型
            # name="head"：分类层的名称
            # **kw：将 kw 字典展开为关键字参数，控制权重初始化方式
            #
            # input_features：
            #   self.rep_size 为 False：features
            #   self.rep_size 为 True：self.width（即 features）
            #   self.rep_size 为正整数：rep_size
            #
            # 输入最后一维：input_features
            # 权重 kernel：[input_features, num_classes]
            # 偏置 bias：[num_classes]，默认初始化为全 0
            # 输出最后一维：num_classes
            #
            # head：分类层模块，x_2d 和 x 共用同一组权重与偏置
            head = nn.Dense(
                self.num_classes,
                dtype=self.dtype_mm,
                name="head",
                **kw,
            )

            # out：保存 ViT 各阶段中间结果的字典
            #
            # seq_len：
            #   pool_type == "tok"：1 + patch_rows * patch_cols
            #   pool_type != "tok"：patch_rows * patch_cols
            #
            # output_features：
            #   self.rep_size 为 False：features
            #   self.rep_size 为 True：features（rep_size = self.width）
            #   self.rep_size 为正整数：rep_size
            #
            # out = {
            #     # 1. Patch Embedding：卷积提取的 Patch 特征
            #     "stem": ...,  # Shape：[batch_size, patch_rows, patch_cols, features]
            #
            #     # 2. Patch Tokens + Positional Embedding（此时尚未添加 CLS Token）
            #     "with_posemb": ...,  # Shape：[batch_size, patch_rows * patch_cols, features]
            #
            #     # 3. 各层 Encoder Block 的中间结果
            #     "encoder": {
            #         "block00": {
            #             "sa":   ...,  # Shape：[batch_size, seq_len, features]
            #             "+sa":  ...,  # Shape：[batch_size, seq_len, features]
            #             "mlp":  ...,  # Shape：[batch_size, seq_len, features]
            #             "+mlp": ...,  # Shape：[batch_size, seq_len, features]
            #         },
            #         "block01": {
            #             "sa":   ...,  # Shape：[batch_size, seq_len, features]
            #             "+sa":  ...,  # Shape：[batch_size, seq_len, features]
            #             "mlp":  ...,  # Shape：[batch_size, seq_len, features]
            #             "+mlp": ...,  # Shape：[batch_size, seq_len, features]
            #         },
            #         # ...
            #     },
            #
            #     # 4. 所有 Encoder Block + 最终 LayerNorm 的输出
            #     # pool_type == "tok" 时包含 CLS Token
            #     "encoded": ...,  # Shape：[batch_size, seq_len, features]
            #
            #     # 5. 池化后的全局特征向量（仅 pool_type != "none" 时存在）
            #     # "map"：Multihead Attention Pooling
            #     # "gap"：Global Average Pooling
            #     # "0"：第 0 个 Patch Token
            #     # "tok"：CLS Token
            #     "head_input": ...,  # Shape：[batch_size, features]
            #
            #     # 6. Patch Tokens 的二维特征图（经过可选 Representation Layer）
            #     # 不进行池化，保留每个 Patch 的空间位置和特征
            #     "pre_logits_2d": ...,  # Shape：[batch_size, patch_rows, patch_cols, output_features]
            #
            #     # 7. 池化及可选 Representation Layer 处理后的特征
            #     # pool_type != "none"：[batch_size, output_features]
            #     # pool_type == "none"：[batch_size, seq_len, output_features]
            #     "pre_logits": ...,
            #
            #     # 8. 每个 Patch 的分类分数（仅 num_classes 非零时存在）
            #     # 使用 Classification Head 将 output_features 映射为 num_classes
            #     # 每个 Patch 对应 num_classes 个原始分类分数（未经 Softmax）
            #     "logits_2d": ...,  # Shape：[batch_size, patch_rows, patch_cols, num_classes]
            #
            #     # 9. 分类层输出的原始分类分数（仅 num_classes 非零时存在）
            #     # pool_type != "none"：[batch_size, num_classes]
            #     # pool_type == "none"：[batch_size, seq_len, num_classes]
            #     # 注意：与 logits_2d 使用同一个 Classification Head，共享权重和偏置
            #     "logits": ...,  # Shape：[batch_size, num_classes] 或 [batch_size, seq_len, num_classes]
            # }

            # 对每个 Patch 的特征向量进行分类映射
            # x_2d → head
            #
            # 输入 x_2d：[batch_size, patch_rows, patch_cols, input_features]
            # 输出 x_2d：[batch_size, patch_rows, patch_cols, num_classes]
            #
            # out["logits_2d"]：保存每个 Patch 对应的 num_classes 个原始分类分数
            # 注意：没有经过 Softmax，因此不是分类概率
            x_2d = out["logits_2d"] = head(x_2d)

            # 对池化后的全局特征向量进行分类映射（pool_type="none" 除外）
            # x → head
            #
            # 输入 x：
            #   pool_type != "none"：[batch_size, input_features]
            #   pool_type == "none"：[batch_size, seq_len, input_features]
            #
            # 输出 x：
            #   pool_type != "none"：[batch_size, num_classes]
            #   pool_type == "none"：[batch_size, seq_len, num_classes]
            #
            # out["logits"]：保存分类层输出的原始分类分数（Logits）
            # 注意：与 x_2d 使用同一个 head，共享权重和偏置
            x = out["logits"] = head(x)

        # return x, out：返回 ViT 的最终输出和各阶段中间结果
        #
        # 1. x：ViT 的最终输出
        #
        #    当 self.num_classes 为正整数时：
        #      pool_type != "none"：[batch_size, num_classes]
        #      pool_type == "none"：[batch_size, seq_len, num_classes]
        #
        #    当 self.num_classes 为 None 或 0 时：
        #      pool_type != "none"：[batch_size, output_features]
        #      pool_type == "none"：[batch_size, seq_len, output_features]
        #
        # 2. out：保存 ViT 各阶段中间结果的字典（dict）
        #
        # seq_len：
        #   pool_type == "tok"：1 + patch_rows * patch_cols
        #   pool_type != "tok"：patch_rows * patch_cols
        #
        # output_features：
        #   self.rep_size 为 False：features
        #   self.rep_size 为 True：features（self.width）
        #   self.rep_size 为正整数：rep_size
        #
        # out = {
        #     # 1. Patch Embedding：卷积提取的 Patch 特征
        #     "stem": ...,  # Shape：[batch_size, patch_rows, patch_cols, features]
        #
        #     # 2. Patch Tokens + Positional Embedding（尚未添加 CLS Token）
        #     "with_posemb": ...,  # Shape：[batch_size, patch_rows * patch_cols, features]
        #
        #     # 3. 各层 Encoder Block 的中间结果
        #     "encoder": {
        #         "block00": {
        #             "sa":   ...,  # Shape：[batch_size, seq_len, features]
        #             "+sa":  ...,  # Shape：[batch_size, seq_len, features]
        #             "mlp":  ...,  # Shape：[batch_size, seq_len, features]
        #             "+mlp": ...,  # Shape：[batch_size, seq_len, features]
        #         },
        #         "block01": {
        #             "sa":   ...,  # Shape：[batch_size, seq_len, features]
        #             "+sa":  ...,  # Shape：[batch_size, seq_len, features]
        #             "mlp":  ...,  # Shape：[batch_size, seq_len, features]
        #             "+mlp": ...,  # Shape：[batch_size, seq_len, features]
        #         },
        #         # ...
        #     },
        #
        #     # 4. 所有 Encoder Block + 最终 LayerNorm 的输出
        #     # pool_type == "tok" 时包含 CLS Token
        #     "encoded": ...,  # Shape：[batch_size, seq_len, features]
        #
        #     # 5. 池化后的全局特征向量（仅 pool_type != "none" 时存在）
        #     # "map"：Multihead Attention Pooling
        #     # "gap"：Global Average Pooling
        #     # "0"：第 0 个 Patch Token
        #     # "tok"：CLS Token
        #     "head_input": ...,  # Shape：[batch_size, features]
        #
        #     # 6. Patch Tokens 的二维特征图（经过可选 Representation Layer）
        #     # 不进行池化，保留每个 Patch 的空间位置和特征
        #     "pre_logits_2d": ...,  # Shape：[batch_size, patch_rows, patch_cols, output_features]
        #
        #     # 7. 池化及可选 Representation Layer 处理后的特征
        #     # pool_type != "none"：[batch_size, output_features]
        #     # pool_type == "none"：[batch_size, seq_len, output_features]
        #     "pre_logits": ...,  # Shape：[batch_size, output_features] 或 [batch_size, seq_len, output_features]
        #
        #     # 8. 每个 Patch 的分类分数（仅 self.num_classes 为正整数时存在）
        #     # 使用 Classification Head 将 output_features 映射为 num_classes
        #     # 每个 Patch 对应 num_classes 个原始分类分数（未经 Softmax）
        #     "logits_2d": ...,  # Shape：[batch_size, patch_rows, patch_cols, num_classes]
        #
        #     # 9. 最终分类分数（仅 self.num_classes 为正整数时存在）
        #     # pool_type != "none"：[batch_size, num_classes]
        #     # pool_type == "none"：[batch_size, seq_len, num_classes]
        #     "logits": ...,  # Shape：[batch_size, num_classes] 或 [batch_size, seq_len, num_classes]
        # }
        #
        # 注意：
        # 1. pool_type == "none" 时，out 中不存在 "head_input"
        # 2. self.num_classes 为 None 或 0 时，out 中不存在 "logits_2d" 和 "logits"
        # 3. pool_type == "tok" 时，out["encoded"] 包含 CLS Token，
        #    但 out["pre_logits_2d"] 和 out["logits_2d"] 只包含 Patch Tokens
        # 4. self.num_classes 为正整数时：x = out["logits"]
        #    否则：x = out["pre_logits"]
        # 5. out 是字典，本身没有 Shape，只有其中的张量具有 Shape
        return x, out

# Module()：ViT 模型的工厂函数（Factory Function）
# 根据 variant 解析模型配置，并创建 _Module 实例
#
# 参数：
#
# 返回：
# _Module 实例（ViT 模型），不是模型的前向计算结果
def Module(
    # num_classes：分类类别数量，默认为 None（不创建分类层）
    num_classes=None, 
    # *：只限制它后面显式声明的普通形参，使这些参数必须通过关键字传递；对 **kw 接收额外关键字参数，以及字典解包传参的行为没有影响
    *, 
    # variant：ViT 模型变体名称，默认为 None，通过 decode_variant() 解析为模型配置字典
    variant=None, 
    # **kw：Keyword Arguments（额外的关键字参数），用于指定或覆盖模型配置，例如 width、depth、num_heads 等
    **kw
):

    # decode_variant(variant)：将模型变体名称解析为配置字典
    # 例如：{"width": 768, "depth": 12, "num_heads": 12, ...}
    #
    # {**decode_variant(variant), **kw}：合并两个配置字典
    # 1. **decode_variant(variant)：展开模型变体的默认配置
    # 2. **kw：展开用户额外传入的配置
    # 3. 如果两个字典存在相同的键，以 **kw 中的值为准
    #
    # _Module(num_classes, **...)：创建 ViT 模型实例
    # num_classes：作为第一个位置参数传入 _Module
    # **：将合并后的配置字典展开为关键字参数

    # Eg：
    # {**decode_variant(variant), **kw} = {
    #     "width": 1152,           # Transformer 隐藏层维度
    #     "depth": 27,             # Transformer Block 层数
    #     "mlp_dim": 4304,         # MLP 中间隐藏层维度
    #     "num_heads": 16,         # Attention Head 数量
    #     "patch_size": (14, 14),  # 每个图像 Patch 的大小
    #     "pool_type": "none",     # 不进行池化，保留所有 Patch Token
    #     "scan": True,            # 使用 nn.scan 构建 Transformer Encoder
    #     "dtype_mm": "bfloat16",  # 矩阵运算的数据类型
    # }
    # 将 {**decode_variant(variant), **kw} 用 ** 解包作为参数传入
    return _Module(num_classes, **{**decode_variant(variant), **kw})

# decode_variant()：解析 ViT 模型变体名称（Model Variant）
# 将字符串形式的模型配置转换为字典（dict）
#
# 参数：
# variant：ViT 模型变体名称，类型为 str 或 None
#   1. None：不指定模型变体，返回空字典 {}
#   2. "B"：只指定模型规模，使用 _Module 中默认的 patch_size
#   3. "B/16"：指定模型规模为 B，Patch 大小为 (16, 16)
#   4. "L/32"：指定模型规模为 L，Patch 大小为 (32, 32)
#
# 返回：
# dict：包含 width、depth、mlp_dim、num_heads 等模型配置
# 如果 variant 包含 "/数字"，还会包含 patch_size
def decode_variant(variant):

    # variant is None：未指定模型变体，返回空字典
    # 后续由 _Module 使用自身的默认配置，或由用户通过 **kw 指定
    if variant is None:
        return {}

    # v：模型规模名称（Variant），初始值为 variant
    # patch：Patch 配置字典，初始为空字典 {}
    #
    # 例如 variant="B/16"：
    #   v = "B/16"
    #   patch = {}
    v, patch = variant, {}

    # "/" in variant：判断模型名称中是否包含 "/"
    # 1. True：同时指定模型规模和 Patch 大小，例如 "B/16"
    # 2. False：只指定模型规模，例如 "B"
    if "/" in variant:

        # str.split("/")：按照 "/" 分割字符串
        #
        # 例如：
        #   variant = "B/16"
        #   variant.split("/") = ["B", "16"]
        #
        #   v = "B"       # 模型规模
        #   patch = "16"  # Patch 大小，当前类型为 str
        v, patch = variant.split("/")

        # int(patch)：将 Patch 大小由字符串转换为整数
        # (int(patch), int(patch))：构造二维 Patch 大小
        #
        # 例如：
        #   patch = "16"
        #   int(patch) = 16
        #   patch = {"patch_size": (16, 16)}
        #
        # patch_size：[patch_height, patch_width]
        # 表示每个 Patch 包含 16 × 16 个像素（以 "B/16" 为例）
        patch = {"patch_size": (int(patch), int(patch))}

    # 根据 v 查找对应的 ViT 模型配置，并返回字典
    #
    # v：模型规模名称，例如 "B"、"L"、"H"
    #
    # 字典索引 [v]：
    #   从配置字典中获取键 v 对应的数值
    #
    # 例如：
    #   {"B": 768, "L": 1024}["B"] = 768
    #
    # 如果 v 不存在于配置字典中，则抛出 KeyError
    return {

        # 1. width：Transformer 隐藏层的特征维度
        #    即每个 Token 的特征维度（Embedding Dimension）
        #
        #    Encoder 输入/输出：[batch_size, seq_len, width]
        #
        #    例如：
        #      v="B" → width=768
        #      v="L" → width=1024
        "width": {
            "mu": 32,
            "Ti": 192,
            "S": 384,
            "M": 512,
            "B": 768,
            "L": 1024,
            "So400m": 1152,
            "H": 1280,
            "g": 1408,
            "g-opt": 1536,
            "G": 1664,
            "G-opt": 1536,
            "e": 1792,
        }[v],

        # 2. depth：Transformer Encoder Block 的层数
        #
        #    例如：
        #      v="B" → depth=12，即包含 12 个 Encoder Block
        #      v="L" → depth=24，即包含 24 个 Encoder Block
        "depth": {
            "mu": 1,
            "Ti": 12,
            "S": 12,
            "M": 12,
            "B": 12,
            "L": 24,
            "So400m": 27,
            "H": 32,
            "g": 40,
            "g-opt": 40,
            "G": 48,
            "G-opt": 48,
            "e": 56,
        }[v],

        # 3. mlp_dim：Encoder Block 中 MLP 的中间隐藏层维度
        #
        #    MLP 的特征维度变化：
        #      [batch_size, seq_len, width]
        #      → [batch_size, seq_len, mlp_dim]
        #      → [batch_size, seq_len, width]
        #
        #    例如：
        #      v="B" → width=768，mlp_dim=3072
        #      v="L" → width=1024，mlp_dim=4096
        "mlp_dim": {
            "mu": 128,
            "Ti": 768,
            "S": 1536,
            "M": 2048,
            "B": 3072,
            "L": 4096,
            "So400m": 4304,
            "H": 5120,
            "g": 6144,
            "g-opt": 6144,
            "G": 8192,
            "G-opt": 8192,
            "e": 15360,
        }[v],

        # 4. num_heads：Multi-Head Self-Attention 中 Query Head 的数量
        #
        #    head_dim = width // num_heads
        #    前提：width 可以被 num_heads 整除
        #
        #    Q、K、V 分头后的典型 Shape：
        #      [batch_size, seq_len, num_heads, head_dim]
        #
        #    例如：
        #      v="B" → width=768，num_heads=12，head_dim=64
        #      v="L" → width=1024，num_heads=16，head_dim=64
        "num_heads": {
            "mu": 2,
            "Ti": 3,
            "S": 6,
            "M": 8,
            "B": 12,
            "L": 16,
            "So400m": 16,
            "H": 16,
            "g": 16,
            "g-opt": 16,
            "G": 16,
            "G-opt": 16,
            "e": 16,
        }[v],

        # pylint:enable=line-too-long
        # 恢复 Pylint 对代码行长度的检查

        # 5. **patch：将 Patch 配置字典展开并合并到返回字典中
        #
        # 1. variant="B"：
        #    patch = {}
        #    不添加 patch_size，使用 _Module 的默认值
        #
        # 2. variant="B/16"：
        #    patch = {"patch_size": (16, 16)}
        #    返回字典中额外包含 "patch_size": (16, 16)
        #
        # **：Dictionary Unpacking（字典解包）
        **patch,
    }