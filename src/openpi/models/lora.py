import math
import re

import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp

import openpi.shared.array_typing as at


@struct.dataclass
class LoRAConfig:

    # W：[a, b] = A：[a, rank], B：[rank, b]
    rank: int

    # 权重参数的初始化器
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)

    # 是否启用 RSLoRA（Rank-Stabilized LoRA）
    # 1. 启用（True）：使用 RSLoRA 缩放方式，alpha / sqrt(rank)
    # 2. 弃用（False）：使用 LoRA 缩放方式，alpha / rank
    rslora: bool = False

    # 缩放参数
    alpha: float = 1.0

    # 缩放系数
    @property # @property 将实例方法转换成通过实例属性的方式访问
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank

    # 设置 A，B 权重 shape 的下标（哪里放rank）
    axes: tuple[int, int] = (-2, -1)

    # 拆分 Einsum 计算表达式的字母
    # 原：(x)[a, b] @ (W)[b, c] = [a, c]
    # 拆：(x)[a, b] @ (A)[b, L] @ (B)[L, c] = [a, c]
    label: str = "L"

# Einsum：基于爱因斯坦求和的线性变换模块，支持 LoRA 微调
# Einstein Summation（爱因斯坦求和）：根据指定的计算表达式执行张量运算
class Einsum(nn.Module):

    # W 的 shape
    shape: tuple[int, ...]

    # W 的 初始化器
    init_fn: nn.initializers.Initializer = nn.initializers.zeros

    # Expert 的 LoRA 配置，默认为 None
    lora_config: LoRAConfig | None = None

    # setup()：定义子模块
    def setup(self):
        # 创建权重 W：[i, o]
        self.w = self.param("w", self.init_fn, self.shape)

        # 海象运算符：先将 self.lora_config 赋值给 config，然后判断 config 是 True of False
        # 创建权重 W_a, W_b
        if config := self.lora_config:

            # shape_a, shape_b = [i, o]
            shape_a, shape_b = list(self.shape), list(self.shape)
            # shape_a = [i, L]
            shape_a[config.axes[1]] = config.rank
            # shape_b = [L, o]
            shape_b[config.axes[0]] = config.rank
            # 创建 W_a：[i, L]
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            # 创建 W_b：[L, o]
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    # Einstein Summation 表达式的拆分：
    # str："bi,io->bo" -> tuple[str, str]：("bi,iL->bL", "bL,Lo->bo")
    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:

        # 检查原始 einsum 公式中是否已经使用了 L，如果已经使用就报错，避免与 LoRA 的 L（rank 维度）冲突
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")

        # re.match("(.*),(.*)->(.*)", eqn)：从字符串开头检查 eqn 是否匹配正则表达式
        # ":="：海象运算符，将匹配结果赋值给 m，整个表达式的结果就是 m 的值
        # m：
        #   1. eqn 满足正则表达式：
        #      m.group(0) = "bi,io->bo"
        #      m.group(1) = "bi"
        #      m.group(2) = "io"
        #      m.group(3) = "bo"
        #      m.groups() = ("bi", "io", "bo")
        #   2. eqn 不满足正则表达式：
        #      m = None
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")

        # lhs（Left-Hand Side）："bi"
        # rhs（Right-Hand Side）："io"
        # output（Output）："bo"
        lhs, rhs, out = m.groups()

        # 确保 LoRA 的 配置 不为空
        assert self.lora_config is not None

        # a_label = i, b_label = o
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        # label = L
        label = self.lora_config.label

        # 拆分后的公式一（"io"->"iL"）：
        # a_rhs = "iL"
        a_rhs = rhs.replace(b_label, label)
        # a_out = "bL"
        a_out = out.replace(b_label, label)
        # eqn_a = "bi,iL->bL"
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        # 拆分后的公式二（"io"->"Lo"）：
        # b_rhs = "Lo"
        b_rhs = rhs.replace(a_label, label)
        # eqn_b = "bL,Lo->bo"
        eqn_b = f"{a_out},{b_rhs}->{out}"

        # eqn_a = "bi,iL->bL"
        # eqn_b = "bL,Lo->bo"
        return eqn_a, eqn_b

    # @nn.compact 允许在 __call__() 方法内部创建参数或子模块。
    # 例如 self.param() 和 nn.Dense() 都可以在该方法中使用。
    @nn.compact
    # 函数调用
    def __call__(self, eqn: str, x):

        # 保留原始输入的类型
        # x：[b, i]
        dtype = x.dtype 

        # result = x @ W：[b, i] @ [i, o] = [b, o]
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        # 海象运算符：先将 self.lora_config 赋值给 config，然后判断 config 是 True of False
        # 创建权重 W_a, W_b
        if config := self.lora_config:

            # eqn_a = "bi,iL->bL"
            # eqn_b = "bL,Lo->bo"
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            # lora = x @ W_a：[b, i] @ [i, L] = [b, L]
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            # lora = (x @ W_a) @ W_b：[b, L] @ [L, o] = [b, o]
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            # result = x @ W + (x @ W_a @ W_b) * (alpha / rank or sqrt(rank))
            # result：[b, o]
            result = result + lora * config.scaling_value

        # result = x @ W | x @ W + (x @ W_a @ W_b) * (alpha / rank or sqrt(rank))
        # result：[b, o]
        return result

# FeedForward：基于门控机制的前馈网络模块，支持 LoRA 微调
class FeedForward(nn.Module):

    # 输入特征的维度
    features: int

    # 隐藏层维度
    hidden_dim: int

    # LoRA 的 配置参数
    lora_config: LoRAConfig | None = None

    # setup()：定义子模块
    def setup(self):

        # W_Gating：[2, features, hidden_dim]
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        # W_Linear：[hidden_dim, features]
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )

        # W_Gating_LoRA：W_Gating 所对应的微调参数
        self.w_gating_lora = None
        # W_Linear_LoRA：W_Linear 所对应的微调参数
        self.w_linear_lora = None

        # 如果 LoRA 配置参数存在
        if self.lora_config:

            # W_Gating_LoRA
            self.w_gating_lora = (
                # W_Gating_LoRA_a：[2, features, rank]
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                # W_Gating_LoRA_b：[2, rank, features]
                self.param("gating_einsum_lora_b", self.lora_config.init_fn, (2, self.lora_config.rank, self.hidden_dim)),
            )
            # W_Linear_LoRA
            self.w_linear_lora = (
                # W_Linear_LoRA_a：[hidden_dim, rank]
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                # W_Linear_LoRA_b：[rank, features]
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )

    # 自定义矩阵乘法运算
    def _dot(
        self, 
        # x：[batch_size, seq_len, features]
        x: at.Array, 
        # W：[features, hidden_dim]
        w: at.Array, 
        # lora_weights：(
        #   W_a：[features, rank],
        #   W_b：[rank, features]
        # )
        lora_weights: tuple[at.Array, at.Array] | None
    ) -> at.Array:

        # base = x @ W
        # x：[batch_size, seq_len, features]
        # W：[features, hidden_dim]
        # base：[batch_size, seq_len, hidden_dim]
        base = jnp.dot(x, w.astype(x.dtype))

        if lora_weights is None:

            # base = x @ W
            return base

        # return x @ W + x @ W_a @ W_b：[batch_size, seq_len, hidden_dim]
        return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))

    # @nn.compact 允许在 __call__() 方法内部创建参数或子模块。
    # 例如 self.param() 和 nn.Dense() 都可以在该方法中使用。
    @nn.compact
    def __call__(self, x):
        
        # 保存输入数据的数据类型
        # x：[batch_size, seq_len, features]
        dtype = x.dtype

        # ff_gate：门控分支输出的中间结果
        # ff_gate：x @ W_Gate | x @ W_Gate + x @ W_Gate_LoRA_a @ W_Gate_LoRA_b
        # ff_gate：[batch_size, seq_len, hidden_dim]
        ff_gate = self._dot(
            # x：[batch_size, seq_len, features]
            x,
            # W_Gate = self.w_gating[0]：[features, hidden_dim]
            self.w_gating[0],
            # W_Gate_LoRA_a = self.w_gating_lora[0][0]：[features, rank]
            # W_Gate_LoRA_b = self.w_gating_lora[1][0]：[rank, features]
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
        )

        # gate_value：门控分支输出
        # gate_value：[batch_size, seq_len, hidden_dim]
        gate_value = nn.gelu(ff_gate)

        # ff1：特征分支输出，与 gate_value 逐元素相乘，得到门控后的特征
        # ff1：x @ W_Value | x @ W_Value + x @ W_Value_LoRA_a @ W_Value_LoRA_b
        # ff1：[batch_size, seq_len, hidden_dim]
        ff1 = self._dot(
            # x：[batch_size, seq_len, features]
            x,
            # W_Value = self.w_gating[1]：[features, hidden_dim]
            self.w_gating[1],
            # W_Value_LoRA_a = self.w_gating_lora[0][1]：[features, rank]
            # W_Value_LoRA_b = self.w_gating_lora[1][1]：[rank, features]
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
        )

        # activations：通过门控后的特征输出
        # *：逐位置相乘
        # activations：[batch_size, seq_len, hidden_dim]
        activations = gate_value * ff1

        # outputs：前馈网络最终输出
        # outputs = activations @ W_Linear + activations @ W_Linear_LoRA_a @ W_Linear_LoRA_b
        # outputs：[batch_size, seq_len, features]
        outputs = self._dot(
            # activations：[batch_size, seq_len, hidden_dim]
            activations, 
            # W_Linear：[hidden_dim, features]
            self.w_linear, 
            # W_Linear_LoRA_a = self.w_linear_lora[0]：[hidden_dim, rank]
            # W_Linear_LoRA_b = self.w_linear_lora[1]：[rank, features]
            self.w_linear_lora
        )

        assert outputs.dtype == dtype
        
        # outputs：[batch_size, seq_len, features]
        return outputs

