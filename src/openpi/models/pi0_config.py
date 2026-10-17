import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

# TYPE_CHECKING：
# 1. 静态文本：True
# 2. 动态执行：False
from typing import TYPE_CHECKING

# 1. 静态类型检查时：导入 Pi0 类，让类型检查工具能够识别 create() 函数的预期返回值类型 Pi0
# 2. 动态执行时：跳过该导入；Pi0 类会在 create() 方法内部导入，供 return Pi0(...) 创建模型实例时使用
if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0

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
#
# frozen=True：
#   1) 限制实例属性的重新赋值，即实例创建后，不能通过普通赋值的方式修改字段
#      例如：config.dtype = "float32" 会触发 FrozenInstanceError
#
#   2) 这不代表字段引用的对象内部也不能修改
#      例如：如果字段的值是一个列表，仍然可以通过 append() 修改列表内容
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):

    # 模型计算使用的数据类型配置
    dtype: str = "bfloat16"

    # PaliGemma 配置
    paligemma_variant: _gemma.Variant = "gemma_2b"

    # Action Expert 配置
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # 每个动作的维度
    action_dim: int = 32

    # 一次预测的动作序列长度
    action_horizon: int = 50

    # 最大 token 长度配置
    max_token_len: int = None 

    # 是否使用 Pi0.5 模型配置（该配置类也可以配置 Pi0.5）
    pi05: bool = False

    # 是否将状态作为离散输入处理
    discrete_state_input: bool = None 

    # PyTorch 编译模式配置
    pytorch_compile_mode: str | None = "max-autotune"

    # __post_init__()：dataclass 自动生成的 __init__() 执行完毕后，自动调用的初始化后处理方法
    def __post_init__(self):

        # 如果没有指定最大 token 长度
        if self.max_token_len is None:

            # object.__setattr__(self, name, value)：通过 Python 内置的属性设置方法，为实例设置属性值
            # 既可以给已有属性重新赋值，也可以设置一个尚不存在的属性
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)

        # 如果没有指定是否将状态作为离散输入处理
        if self.discrete_state_input is None:

            # object.__setattr__(self, name, value)：通过 Python 内置的属性设置方法，为实例设置属性值
            # 既可以给已有属性重新赋值，也可以设置一个尚不存在的属性
            object.__setattr__(self, "discrete_state_input", self.pi05)

        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                # 1. 默认模式：使用常规编译优化，兼顾编译开销与运行性能
                "default",
                # 2. 减少运行开销：重点降低 Python 调度等开销，可能使用 CUDA Graph
                "reduce-overhead",
                # 3. 最大化自动调优：更积极地选择高性能运算实现，通常会增加编译和调优开销
                "max-autotune",
                # 4. 最大化自动调优（不使用 CUDA Graph）：保留自动调优策略，同时禁用 CUDA Graph 优化
                "max-autotune-no-cudagraphs",
            ]

    # @property：将方法转换为属性，使其可以像访问成员变量一样访问，无需添加 ()
    @property
    # @override：标记该方法重写（Override）了父类中的方法，供类型检查工具检查
    @override
    # 获取当前模型的类型（Model Type）  
    def model_type(self) -> _model.ModelType:
        
        # 使用 Pi0.5 模型
        if self.pi05:

            # 返回 Pi0.5 模型类型
            return _model.ModelType.PI05

        # 返回 Pi0 模型类型    
        return _model.ModelType.PI0

    # @override：标记该方法重写（Override）了父类中的方法，供类型检查工具检查
    @override
    # 创建 Pi0 模型
    def create(
        self, 
        # 随机数密钥
        rng: at.KeyArrayLike
    ) -> "Pi0": # 动态执行时，TYPE_CHECKING=False，防止 Pi0 类解析失败，故加""号，延迟解析
        
        from openpi.models.pi0 import Pi0

        # 创建 PI0 模型
        return Pi0(
            self, 
            
            # 根据 JAX 的随机数密钥 rng，创建一个用于管理随机数密钥的对象 rngs，供后续模型初始化等操作使用
            rngs=nnx.Rngs(rng)
        )

    # @override：标记该方法重写（Override）了父类中的方法，供类型检查工具检查
    @override
    # inputs_spec（Input Specification）：返回模型输入的抽象数据规格
    # Observation 和 Actions 中的张量均由 jax.ShapeDtypeStruct 描述，仅包含 shape 和 dtype，没有真实数据    
    def inputs_spec(
        self, 
        *, 
        batch_size: int = 1
    ) -> tuple[_model.Observation, _model.Actions]:

        # 图像输入数据规格：[batch_size, heigh(224), width(224), channels(3)]
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        # 图像掩码数据规格：[batch_size]
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        # with at.disable_typechecking()：临时禁用运行时类型检查
        # with：仅在代码块内部禁用，退出后恢复原来的类型检查状态
        #
        # 原因：
        # 1. @at.typecheck：启用运行时类型检查（Runtime Type Checking）
        #    1. 只能直接装饰类和函数，不能直接装饰普通变量或类型别名
        #    2. 装饰类：创建实例时，检查字段是否符合类型注解，
        #       包括数组类型（Array Type）、数据类型（dtype）和形状（shape）
        #    3. 装饰函数：调用时检查输入参数，返回时检查返回值是否符合类型注解，
        #       包括数组类型（Array Type）、数据类型（dtype）和形状（shape）
        #    注意：具体检查内容取决于类型注解及检查器的实现
        #
        # 2. _model.Observation 类使用了 @at.typecheck 装饰器，
        #    因此创建 Observation 实例时，会检查字段是否符合类型注解。
        #
        # 3. Observation 的 ArrayT 约束为：
        #    jax.Array | torch.Tensor | np.ndarray
        #    但 inputs_spec() 使用的是 jax.ShapeDtypeStruct，
        #    它只描述数组的 shape 和 dtype，不包含真实数组数据，
        #    也不属于上述 ArrayT 允许的数组类型。
        #
        # 4. 因此，创建 Observation 实例时，需要临时禁用运行时类型检查，
        #    以允许使用 jax.ShapeDtypeStruct 构建抽象输入规格。
        with at.disable_typechecking():

            # 创建 机器人观测数据实例
            observation_spec = _model.Observation(

                # 图像输入
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                # 图像掩码
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },

                # 机器人状态：[batch_size, action_dim]
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),

                # 语言 token：[batch_size, seq_len]
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),

                # 语言 token 掩码：[batch_size, seq_len]
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )

        # 此时创建 action_spec 时没有调用 _model.Action 类别名，而且类别名不使用 @at.typecheck 装饰器，所以不会检查返回值的类型注解，包括数组类型（Array Type）、数据类型（dtype）和形状（shape），故不需要临时禁用运行时类型检查
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        # 返回模型输入的抽象数据规格
        return observation_spec, action_spec

    # get_freeze_filter()：生成模型参数冻结过滤器
    #
    # 1. LoRA（Low-Rank Adaptation，低秩适配）的目标：
    #    1. 冻结预训练模型的原始权重 W，不直接更新这些大规模参数
    #    2. 引入少量可训练的低秩参数 A、B，通过 ΔW = BA 近似原始权重的更新
    #    3. 前向计算使用 W + ΔW，而训练时只更新 A、B
    #    4. 以少量可训练参数的更新，替代对大规模原始权重的直接更新
    #    5. 减少可训练参数量、优化器状态占用和训练成本
    #
    # 2. 在 π0 中：
    #    1. PaliGemma 中的 Gemma 可以使用 LoRA
    #    2. Action Expert 中的 Gemma 也可以使用 LoRA
    #    3. 哪部分使用 LoRA，就冻结该部分的原始权重，保留 LoRA 参数可训练
    #    4. 哪部分不使用 LoRA，就不冻结该部分的原始权重
    #
    # 3. 本函数的作用：
    #    根据 PaliGemma 和 Action Expert 的 LoRA 配置，生成模型参数路径过滤器
    #    用于匹配需要冻结的原始模型参数，同时排除需要训练的 LoRA 参数，保证 LoRA 参数可以正常训练
    #
    # 返回值：nnx.filterlib.Filter，模型参数冻结过滤器
    def get_freeze_filter(self) -> nnx.filterlib.Filter:

        # filters：过滤器列表，保存多个模型参数路径过滤条件
        filters = []

        # has_lora：标记当前模型是否使用 LoRA
        # 只要 PaliGemma 或 Action Expert 中至少一个使用 LoRA，就设置为 True
        has_lora = False

        # π0（Vision-Language-Action Model，视觉-语言-动作模型）
        # │
        # ├── PaliGemma（Vision-Language Model，视觉语言模型）
        # │   ├── SigLIP Vision Encoder（视觉编码器）
        # │   └── Gemma（语言模型）
        # │
        # └── Action Expert（动作专家）
        #     └── Gemma-style Transformer（动作建模网络）

        # nnx_utils.PathRegex()：生成通过正则表达式（Regular Expression）
        # 匹配模型参数路径的过滤器

        # 1. gemma_params_filter：模型参数路径过滤器
        #    1. 匹配以下两部分的模型参数路径：
        #       1. PaliGemma 中 Gemma 的模型参数路径
        #       2. Action Expert 中 Gemma 的模型参数路径
        #    2. ".*llm.*"：匹配任意位置包含 "llm" 的模型参数路径
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")

        # 2. action_expert_params_filter：Action Expert 的模型参数路径过滤器
        #    1. 只匹配 Action Expert 中 Gemma 的模型参数路径
        #    2. ".*llm.*_1.*"：匹配包含 "llm"，且其后出现 "_1" 的模型参数路径
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")

        # 情况一：PaliGemma 中的 Gemma 使用 LoRA
        if "lora" in self.paligemma_variant:

            # 添加 gemma_params_filter：
            # 匹配 PaliGemma 中 Gemma 和 Action Expert 中 Gemma 的模型参数路径
            filters.append(
                gemma_params_filter,
            )

            # 如果 Action Expert 中的 Gemma 不使用 LoRA
            if "lora" not in self.action_expert_variant:

                # nnx.Not()：对过滤条件取反
                # action_expert_params_filter：匹配 Action Expert 中 Gemma 的模型参数路径
                # nnx.Not(action_expert_params_filter)：匹配不属于 Action Expert 中 Gemma 的参数路径
                # 目的：从冻结范围中排除 Action Expert 中 Gemma 的参数，使其保持可训练
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )

            # 标记当前模型使用了 LoRA
            # 无论 Action Expert 中的 Gemma 是否使用 LoRA，此处均为 True
            has_lora = True

        # 情况二：PaliGemma 中的 Gemma 不使用 LoRA，而 Action Expert 中的 Gemma 使用 LoRA
        elif "lora" in self.action_expert_variant:

            # 添加 action_expert_params_filter：
            # 只匹配 Action Expert 中 Gemma 的模型参数路径
            # 目的：冻结 Action Expert 中 Gemma 的原始模型参数
            # 同时保留 PaliGemma 中 Gemma 的原始参数可训练
            filters.append(
                action_expert_params_filter,
            )

            # 标记当前模型使用了 LoRA
            has_lora = True

        # 如果 PaliGemma 中的 Gemma 或 Action Expert 中的 Gemma 至少有一个使用 LoRA
        if has_lora:

            # nnx_utils.PathRegex(".*lora.*")：匹配包含 "lora" 的模型参数路径
            # nnx.Not()：对过滤条件取反，匹配不包含 "lora" 的模型参数路径
            # 目的：从冻结范围中排除 LoRA 参数，使 LoRA 参数保持可训练
            # 最终只冻结使用 LoRA 的 Gemma 模型中的原始权重
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )

        # 如果过滤器列表为空，说明两部分 Gemma 均不使用 LoRA
        if not filters:

            # nnx.Nothing：不匹配任何模型参数路径的过滤器
            # 表示不通过该过滤器冻结任何模型参数
            return nnx.Nothing

        # *filters：将过滤器列表展开为多个位置参数
        # nnx.All(*filters)：将所有过滤条件组合为一个过滤器，只有同时满足全部过滤条件的模型参数路径才会被匹配
        # 返回值：多个不同过滤器组合后的模型参数冻结过滤器，而不是模型参数本身
        return nnx.All(*filters)