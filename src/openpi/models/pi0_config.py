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

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
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

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
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
