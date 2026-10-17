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

    def get_freeze_filter(self) -> nnx.filterlib.Filter:

        filters = []
        has_lora = False

        # π0（Vision-Language-Action Model）
        # │
        # ├── PaliGemma（VLM）
        # │   ├── SigLIP Vision Encoder（视觉编码器）
        # │   └── Gemma 2B（语言模型）
        # │
        # └── Action Expert（动作专家）
        #     └── Gemma-style Transformer（约 300M 参数）
        
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")

        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
