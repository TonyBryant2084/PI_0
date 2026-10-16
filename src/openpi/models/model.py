import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

# 创建一个名为 "openpi" 的日志记录器
# 后续可通过 logger.info() / warning() /error() 输出日志
logger = logging.getLogger("openpi")

# 类型数组，兼容三种张量类型
# 1. 第一个 ArrayT 是 类型变量对象
# 2. 第二个 ArrayT 是 类型变量对象内部登记的名字
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)

# 定一个枚举类型，用来表示 OpenPI 支持哪些模型
# enum.Enum 是 Python 标准库提供的枚举基类
class ModelType(enum.Enum):

    # 1. Pi0
    PI0 = "pi0"
    # 2. PI0_FAST
    PI0_FAST = "pi0_fast"
    # 3. PI05
    PI05 = "pi05"


# 定一个元组表示 OpenPI 模型需要接受哪些图像输入
IMAGE_KEYS = (
    # 1. 基座 第 0 个 RGB相机
    "base_0_rgb",
    # 2. 左手手腕 第 0 个 RGB 相机
    "left_wrist_0_rgb",
    # 3. 右手手腕 第 0 个 RGB 相机
    "right_wrist_0_rgb",
)

# 图像分辨率
IMAGE_RESOLUTION = (224, 224)

# @at.typecheck 是一个类型检查装饰器，用于检查函数调用是否符合声明的类型要求。
@at.typecheck

# @struct.dataclass：Flax 提供的数据类装饰器
# 1. 自动生成 __init__、__repr__、__eq__ 等方法
#    __init__：将类中声明的数据类字段初始化为实例属性
# 2. 实例属性默认不可重新赋值，使用 obj.replace() 创建更新后的新对象
# 3. 自动注册为 JAX PyTree，支持 jax.tree.map() 等操作
# 4. struct.field(pytree_node=False)：将字段设为静态元数据，不作为 PyTree 叶子节点
#    例如：name: str = struct.field(pytree_node=False)
@struct.dataclass

# Observation 是一个机器人观测数据类（Observation Data Class），用于将图像、机器人状态、语言指令等输入数据统一封装成一个对象
# Generic[ArrayT]：声明 Observation 是泛型类，类中使用 ArrayT 的类型注解由指定的类型决定
class Observation(Generic[ArrayT]):

    # 图像：[batch_size, heigh, width, channels] 
    images: dict[str, at.Float[ArrayT, "*b h w c"]]

    # 图像掩码：[batch_size]
    # 用 True or False 表示每批图像是否有效
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]

    # 机器人状态：[batch_size, state_dimension]
    state: at.Float[ArrayT, "*b s"]

    # 语言 token：[batch_size, seq_len]
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None

    # 语言 token 掩码：[batch_size, seq_len]
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # 6. pi0-fast 模型专用字段。
    # 现在阅读普通 pi0 时可以暂时跳过。
    # ------------------------------
    # Token 自回归 Mask。
    # FAST 自回归模型使用。
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # ------------------------------
    # Token Loss Mask。
    # FAST 自回归模型使用。
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    # 类方法：可以通过类名或实例调用，无需创建实例即可通过类名调用
    @classmethod

    # 字典 -> Observation 实例
    def from_dict(
        cls, 
        # PyTree（Python Tree）：JAX 支持的树形数据结构
        # 1. 由容器节点（Container Nodes）和叶子节点（Leaf Nodes）组成，也可以只有一个叶子节点
        # 2. 容器节点：dict、list、tuple 等，支持多层嵌套
        # 3. 叶子节点：jax.Array、int、float 等实际数据
        # 4. 字典（dict）：
        #    - dict：容器节点
        #    - Key：结构信息，不属于叶子节点
        #    - Value：可以是容器节点，也可以是叶子节点
        # 5. PyTree 可以嵌套，也可以不嵌套
        # 6. PyTree[ArrayT]：约束叶子节点应为 ArrayT 类型
        # 
        # data 举例：
        # data = {
        #     "image": {
        #         "base_0_rgb": ...,         # [*b, h, w, c]
        #         "left_wrist_0_rgb": ...,   # [*b, h, w, c]
        #         "right_wrist_0_rgb": ...,  # [*b, h, w, c]
        #     },
        #
        #     "image_mask": {
        #         "base_0_rgb": ...,         # [*b]
        #         "left_wrist_0_rgb": ...,   # [*b]
        #         "right_wrist_0_rgb": ...,  # [*b]
        #     },
        #
        #     "state": ...,                  # [*b, s]
        #
        #     "tokenized_prompt": ...,       # [*b, l]
        #
        #     "tokenized_prompt_mask": ...,  # [*b, l]
        #
        #     "token_ar_mask": ...,          # [*b, l]
        #
        #     "token_loss_mask": ...,        # [*b, l]
        # }
        data: at.PyTree[ArrayT]
    ) -> "Observation[ArrayT]": # 因为此时 Observation 类尚未定义完成，所以用""号进行延迟解析

        # 1. 语言 token 和 语言 mask 应该同时存在
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")

        # 2. 图像如果是 uint8 [0, 255]，则转成 float32 [-1, 1]
        for key in data["image"]:
            # 1) 当前 key 对应的图像数据类型为 np.uint8 [0, 255]
            if data["image"][key].dtype == np.uint8:
                # 则转成 np.float32 [-1, 1]
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            # 2) 当前 key 对应的图像数据类型为 torch.uint8 [0, 255]
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                # 则转成 torch.float32 [-1, 1]
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0

        # 创建实例并返回
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    # Observation 实例 -> dict
    def to_dict(self) -> at.PyTree[ArrayT]:

        # 将具备 dataclass 结构的 Observation 对象通过 dataclasses.asdict() 转换成普通字典
        result = dataclasses.asdict(self)

        # result.pop("images")：删除 result 字典中 Key 为 "images" 的键值对，并返回对应的 Value
        # result["image"] = ...：将返回的 Value 以新 Key "image" 存入 result 字典
        # 整体作用：将字典中的 Key 从 "images" 重命名为 "image"，Value 保持不变
        result["image"] = result.pop("images")

        # result.pop("image_masks")：删除 result 字典中 Key 为 "image_masks" 的键值对，并返回对应的 Value
        # result["image_mask"] = ...：将返回的 Value 以新 Key "image_mask" 存入 result 字典
        # 整体作用：将字典中的 Key 从 "image_masks" 重命名为 "image_mask"，Value 保持不变
        result["image_mask"] = result.pop("image_masks")

        # 返回结果字典
        return result

# Actions：Type Alias（类型别名），用于定义动作张量的类型
# 1. 数组类型：ArrayT，可以是 jax.Array、torch.Tensor 或 np.ndarray（取决于 ArrayT 的具体类型）
# 2. 数组形状（Shape）：[*b, ah, ad]，其中：
#    *b：Batch Dimensions，零个或多个批次维度
#    ah：Action Horizon，动作序列长度
#    ad：Action Dimension，每个动作的维度
# 3. 元素数据类型：Float，要求数组元素是浮点数类型，例如 float32、float64，并非只能是 float32
Actions = at.Float[ArrayT, "*b ah ad"]

# 预处理机器人观测数据
def preprocess_observation(
    # 随机数密钥
    rng: at.KeyArrayLike | None,
    # 机器人观测数据实例
    observation: Observation,
    # 关键字参数分隔符 *
    # 1. * 号前面的参数可以按照 "位置传参"，也可以按照 "关键字传参"
    # 2. * 号后面的参数必须按照 "关键字传参"
    *,
    # 是否启用训练阶段的数据增强
    # 1. train = True
    #    非腕部相机：随机裁剪 95% + 恢复原尺寸 + 随机旋转 ±5° + 颜色抖动
    #    腕部相机：只做颜色抖动
    #    目的：增加训练数据的多样性，提高模型对不同画面变化的适应能力
    # 2. train = False
    #    不进行随机数据增强，只进行必要的尺寸处理
    train: bool = False,
    # 要处理哪些图像输入
    image_keys: Sequence[str] = IMAGE_KEYS,
    # 图像分辨率
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:

    # 检查要处理的图像名称是否都存在于机器人观测数据的图像字典中
    #
    # 1. set(image_keys)：将元组转换为集合，方便判断集合之间的包含关系
    #
    #    元组（Tuple）：有序、可重复、不可变、支持索引
    #    image_keys = (
    #        "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb",
    #    )
    #
    #    集合（Set）：无序、不可重复、可变、不支持索引
    #    set(image_keys) = {
    #        "base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb",
    #    }
    #
    # 2. A.issubset(B)：判断集合 A 是否为集合 B 的子集
    #    即判断 A 中的所有元素是否都存在于 B 中
    #    如果全部存在，返回 True，否则返回 False
    #
    #    observation.images：图像字典，在字典迭代时默认遍历 Key
    #
    #    set(image_keys).issubset(observation.images)：
    #    判断 set(image_keys) 中的所有图像名称是否都存在于 observation.images 的 Key 中
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    # 机器人状态数据的批次
    # 1. shape[:-1]：维度中除了最后一维其他都要
    # 2. observation.state.shape = (batch_size, state_dim)
    # 3. observation.state.shape[:-1] = (batch_size,)
    batch_shape = observation.state.shape[:-1]

    # 经过预处理后的图像
    out_images = {}

    # 遍历每个相机
    for key in image_keys:

        # 从观察数据中取出该相机对应的图像
        # image：[batch_size, heigh, width, channels]
        image = observation.images[key]

        # 若输入图像的分辨率和规定分辨率不同
        # 1. image.shape = (batch_size, heigh, width, channels)
        # 2. image.shape[1:3] = (heigh, width)
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")

            # 先将图像等比例缩放，再通过 Padding（填充）补齐不足的部分，使图像达到指定分辨率，同时保持原始图像内容的宽高比不变
            # 输入/输出 image：[batch_size, heigh, width, channels]
            image = image_tools.resize_with_pad(image, *image_resolution)

        # 训练模式：train = True
        # 1. 非腕部相机：随机裁剪 95% + 恢复原尺寸 + 随机旋转 ±5° + 颜色抖动
        # 2. 腕部相机：只做颜色抖动
        # 3. 目的：增加训练数据的多样性，提高模型对不同画面变化的适应能力
        if train:   

            # augmax 的图像增强操作需要将图像处于 [0, 1]，则要从 [-1, 1] -> [0, 1]
            image = image / 2.0 + 0.5

            # 创建一个空列表，用来存放后面要执行的图像变换操作
            transforms = []

            # 非腕部相机：随机裁剪 95% + 恢复原尺寸 + 随机旋转 ±5° + 颜色抖动
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    # 1. 随机裁剪 95%（需要随机数密钥）
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    # 2. 将裁剪后的图像恢复到原来的尺寸
                    augmax.Resize(width, height),
                    # 3. 随机旋转 ±5°（需要随机数密钥）
                    augmax.Rotate((-5, 5)),
                ]

            # 腕部相机：只做颜色抖动
            transforms += [
                # 颜色抖动：随机改变亮度、对比度和饱和度（需要随机数密钥）
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]

            # 将随机数密钥 rng 拆分为 batch_size 个随机数子密钥
            # rng：原始随机数密钥
            # image.shape[0]：batch_size，批次中的图像数量
            # sub_rngs：随机数子密钥数组，Shape = (batch_size,)
            sub_rngs = jax.random.split(rng, image.shape[0])

            # 对图像批次中的每张图像分别执行对应的增强操作
            # 1. 将多个图像增强操作组合成一个增强函数：
            #    augment_fn = augmax.Chain(*transforms)
            #
            # 2. jax.vmap()：Vectorizing Map（向量化映射）
            #    接收增强函数，将其转换为支持批量处理的函数：
            #    batch_augment_fn = jax.vmap(augment_fn)
            #
            # 3. 传入随机数子密钥和图像批次，执行图像增强：
            #    sub_rngs：随机数子密钥数组，Shape = (batch_size,)
            #    image：图像批次，Shape = (batch_size, height, width, channels)
            #    每张图像使用一个独立的随机数子密钥
            #    image = batch_augment_fn(sub_rngs, image)
            #
            # 4. 输出：
            #    image：增强后的图像批次
            #    Shape = (batch_size, height, width, channels)
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # 恢复到 [-1, 1]
            image = image * 2.0 - 1.0

        # 接收预处理后的图像
        out_images[key] = image

    # 图像掩码收集
    out_masks = {}
    
    # 遍历预处理后图像所对应的相机
    for key in out_images:

        # 图像掩码中没有关于该相机的数据
        if key not in observation.image_masks:

            # 默认该相机所对应的图像均有效，图像掩码均为 True(1)
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)

        # 图像掩码中有关于该相机的数据
        else:

            # 使用该图像掩码
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    # 返回预处理后的机器人观测数据
    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )

# @dataclasses.dataclass：Python 标准库提供的数据类装饰器
# 1. 自动生成 __init__、__repr__、__eq__ 等方法
#    __init__：根据类中声明的字段初始化实例属性
#    __repr__：生成对象的字符串表示
#    __eq__：比较两个对象的字段值是否相等
# 2. frozen=True：创建冻结数据类，实例创建后不能重新赋值或删除字段
#    如果尝试修改字段，会抛出 dataclasses.FrozenInstanceError
# 3. frozen=True 只限制字段重新赋值，不保证字段内部的可变对象不可修改
@dataclasses.dataclass(frozen=True)
# 模型配置基类
# abc.ABC：Abstract Base Classes 抽象基类
class BaseModelConfig(abc.ABC):

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)

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
# abc.ABC：Abstract Base Classes 抽象基类
class BaseModel(nnx.Module, abc.ABC):

    # 动作维度：一个动作需要多少个部分共同完成
    action_dim: int

    # 动作序列长度：一次预测产生的动作数量
    action_horizon: int

    # 最大 token 数：语言提示可最多包含的 token 数
    max_token_len: int

    # 抽象类
    @abc.abstractmethod
    # 根据观测信息和真实动作计算损失
    def compute_loss(
        self,
        rng: at.KeyArrayLike,              # JAX 随机数密钥，用于生成随机数
        observation: Observation,          # 视觉语言模型的输入
        actions: Actions,                  # 真实动作数据
        *,                                 # 前者参数位置传参和关键字传参都可以，后者只能关键字传参
        train: bool = False,               # 是否处于训练模式
    ) -> at.Float[at.Array, "*b ah"]: ...  # 返回浮点类型损失张量，shape：[batch_size, action_horizon]，*b 表示批次维度可以有零个、一个或多个，具体有多少个维度由实际输入决定

    # 抽象类
    @abc.abstractmethod
    # 根据观测信息生成动作序列
    def sample_actions(
        self,  
        rng: at.KeyArrayLike,      # JAX 随机数密钥，用于生成随机数
        observation: Observation,  # 视觉语言模型的输入 
        **kwargs                   # 接收额外的关键字参数
    ) -> Actions: ...              # 返回模型生成的动作序列


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
