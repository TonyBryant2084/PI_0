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
    y, x = jnp.mgrid[:h, :w]

    # width 要是 4 的倍数
    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    
    # omega：不同特征维度对应的角频率（Angular Frequency）
    # omega = [0, 1/(width/4-1), ..., (width/4-2)/(width/4-1), 1]
    # omega = [width/4]
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    # omega = [1/(10000^(0)), ..., 1/(10000^1)]
    omega = 1.0 / (temperature**omega)


    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
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
        return posemb_sincos_2d(*seqshape, width, dtype=dtype)
    raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        """Applies Transformer MlpBlock module."""
        inits = {
            "kernel_init": nn.initializers.xavier_uniform(),
            "bias_init": nn.initializers.normal(stddev=1e-6),
        }

        _, _, d = x.shape  # n,l,d
        x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        return nn.Dense(d, dtype=self.dtype_mm, **inits)(x)


class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}
        x = sharding.activation_sharding_constraint(x)
        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["sa"] = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
            dtype=self.dtype_mm,
        )(y, y)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+sa"] = x + y

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            dtype_mm=self.dtype_mm,
        )(y, deterministic)
        y = sharding.activation_sharding_constraint(y)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        x = out["+mlp"] = x + y
        x = sharding.activation_sharding_constraint(x)
        return x, out


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    depth: int
    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    scan: bool = False
    remat_policy: str = "nothing_saveable"
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x, deterministic=True):  # noqa: FBT002
        out = {}

        if self.scan:
            block = nn.remat(
                Encoder1DBlock,
                prevent_cse=False,
                static_argnums=(2,),  # 0=self, 2=deterministic
                policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
            )
            x, scan_out = nn.scan(
                block,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=nn.broadcast,
                length=self.depth,
            )(
                name="encoderblock",
                dtype_mm=self.dtype_mm,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )(x, deterministic)
            for lyr in range(self.depth):
                out[f"block{lyr:02d}"] = jax.tree.map(lambda o, lyr=lyr: o[lyr], scan_out)
        else:
            # Input Encoder
            for lyr in range(self.depth):
                block_cur = Encoder1DBlock(
                    name=f"encoderblock_{lyr}",
                    dtype_mm=self.dtype_mm,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                )
                x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
            out["pre_ln"] = x  # Alias for last block, but without the number in it.

        return nn.LayerNorm(name="encoder_norm", dtype=self.dtype_mm)(x), out


class MAPHead(nn.Module):
    """Multihead Attention Pooling."""

    mlp_dim: int | None = None  # Defaults to 4x input dim
    num_heads: int = 12
    dtype_mm: str = "float32"

    @nn.compact
    def __call__(self, x):
        n, _, d = x.shape  # n,l,d
        probe = self.param("probe", nn.initializers.xavier_uniform(), (1, 1, d), x.dtype)
        probe = jnp.tile(probe, [n, 1, 1])

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dtype=self.dtype_mm,
            kernel_init=nn.initializers.xavier_uniform(),
        )(probe, x)

        y = nn.LayerNorm(dtype=self.dtype_mm)(x)
        x = x + MlpBlock(mlp_dim=self.mlp_dim, dtype=self.dtype_mm)(y)
        return x[:, 0]


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

    # rep_size：Representation Size（表示层输出维度）
    # 1. int：创建指定输出维度的表示层，例如 rep_size=512，则表示层输出维度为 512
    # 2. bool：
    #   1) True：创建输出维度为 width 的表示层
    #   2) False：不创建额外的表示层
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
        # out["stem"] = nn.Conv(...)(image)：将结果存入 out dict，key 为 "stem"
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

        # Add posemb before adding extra token.
        x = out["with_posemb"] = x + get_posemb(self, self.posemb, (h, w), c, "pos_embedding", jnp.float32)

        if self.pool_type == "tok":
            cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
            x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

        n, _, c = x.shape  # n,l,d
        x = nn.Dropout(rate=self.dropout)(x, not train)

        # Kevin edit: now cast back to dtype_mm (potentially half precision)
        x = x.astype(self.dtype_mm)

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
        encoded = out["encoded"] = x

        if self.pool_type == "map":
            x = out["head_input"] = MAPHead(
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dtype=self.dtype_mm,
            )(x)
        elif self.pool_type == "gap":
            x = out["head_input"] = jnp.mean(x, axis=1)
        elif self.pool_type == "0":
            x = out["head_input"] = x[:, 0]
        elif self.pool_type == "tok":
            x = out["head_input"] = x[:, 0]
            encoded = encoded[:, 1:]
        elif self.pool_type == "none":
            pass
        else:
            raise ValueError(f"Unknown pool type: '{self.pool_type}'")

        x_2d = jnp.reshape(encoded, [n, h, w, -1])

        if self.rep_size:
            rep_size = self.width if self.rep_size is True else self.rep_size
            hid = nn.Dense(rep_size, dtype=self.dtype_mm, name="pre_logits")
            # NOTE: In the past we did not include tanh in pre_logits.
            # For few-shot, it should not matter much, as it whitens anyways.
            x_2d = nn.tanh(hid(x_2d))
            x = nn.tanh(hid(x))

        out["pre_logits_2d"] = x_2d
        out["pre_logits"] = x

        if self.num_classes:
            kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
            head = nn.Dense(self.num_classes, dtype=self.dtype_mm, name="head", **kw)
            x_2d = out["logits_2d"] = head(x_2d)
            x = out["logits"] = head(x)

        return x, out


def Module(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name  # noqa: N802
    """Factory function, because linen really don't like what I'm doing!"""
    return _Module(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v, patch = variant, {}
    if "/" in variant:
        v, patch = variant.split("/")
        patch = {"patch_size": (int(patch), int(patch))}

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
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
        **patch,
    }
