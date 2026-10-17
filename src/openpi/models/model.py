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

    # 动作维度
    action_dim: int
    # 动作序列长度
    action_horizon: int
    # 语言 token 最大长度
    max_token_len: int

    # @property：将方法转换为属性，使其可以像访问成员变量一样访问，无需添加 ()
    @property
    # @abc.abstractmethod：声明抽象方法，要求具体子类实现
    @abc.abstractmethod
    # model_type：返回模型类型（ModelType）
    def model_type(self) -> ModelType: ...
        
    # @abc.abstractmethod：声明抽象方法，要求具体子类实现
    @abc.abstractmethod
    # create：根据随机数密钥 rng 创建并返回 BaseModel 模型实例
    def create(self, rng: at.KeyArrayLike) -> "BaseModel": ...

    # load：将已有模型参数 params 加载到模型中，并返回加载参数后的 BaseModel 实例
    def load(
        self, 
        # 待加载的模型参数（Parameters），通常为 PyTree 结构
        params: at.Params,
        *, 
        # 是否移除 params 中模型不需要的额外参数，默认 True
        remove_extra_params: bool = True
    ) -> "BaseModel":

        # 创建抽象模型
        # 通过模型创建函数 self.create 和随机数密钥 rng，抽象创建一个具有完整模型结构的抽象模型，其中权重参数只有 shape 和 dtype 信息，没有真实的数值数据
        model = nnx.eval_shape(self.create, jax.random.key(0))

        # nnx.split：将模型拆分为模型结构和模型状态，支持普通模型和抽象模型
        # graphdef：模型结构（Graph Definition），不包含真实参数数值
        # state：模型状态（State），保存模型的权重等状态变量
        #   1. 普通模型：包含真实权重数值及其 shape、dtype
        #   2. 抽象模型：仅包含权重等变量的 shape、dtype，没有真实数值
        graphdef, state = nnx.split(model)

        # 如果 remove_extra_params=True，则移除 params 中模型不需要的额外参数
        if remove_extra_params:
            # 输入：
            #   1. state.to_pure_dict()：将模型状态转换为普通字典形式的 PyTree，作为预期参数结构
            #   2. params：待加载的真实模型参数
            # ocp.transform_utils.intersect_trees：根据两个 PyTree 的共同路径，保留 params 中与 state 对应的参数
            # 输出：params：筛选后的模型参数，移除了模型不需要的额外参数
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)

        # at.check_pytree_equality：检查待加载参数与抽象模型的参数是否匹配
        # 注意：同时也会检查 PyTree 的结构是否一致，不匹配时抛出异常
        at.check_pytree_equality(
            # expected：抽象模型的预期参数，包含 PyTree 结构及各参数的 shape、dtype
            expected=state.to_pure_dict(), 
            # got：待加载的真实模型参数 params
            got=params, 
            # check_shapes=True：检查对应参数的 shape 是否一致
            check_shapes=True, 
            # check_dtypes=False：不检查对应参数的 dtype 是否一致
            check_dtypes=False
        )
   
        # state.replace_by_pure_dict(params)：使用真实模型参数 params 替换 state 中的抽象参数
        # state：替换后的模型状态，包含真实权重数值及其 shape、dtype
        state.replace_by_pure_dict(params)

        # nnx.merge：将模型结构 graphdef 和真实模型状态 state 合并，重建完整模型
        # 返回值：包含真实权重参数的 BaseModel 模型实例，可用于后续训练或推理
        return nnx.merge(graphdef, state)

    # load_pytorch：根据训练配置创建 PyTorch 版本的 π0 模型，加载预训练权重并返回模型实例
    def load_pytorch(
        self, 
        # 训练配置（Training Configuration），包含模型配置等信息
        train_config, 
        # 预训练权重文件路径（Weight Path）
        weight_path: str
    ):
        # # logger.info：将训练配置信息输出到日志
        logger.info(f"train_config: {train_config}")

        # PI0Pytorch：根据 train_config.model 创建 PyTorch 版本的 π0 模型
        # 1. train_config.model：模型配置
        # 2. model：包含真实的初始化权重（通常是随机初始化）的模型
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)

        # safetensors.torch.load_model：从权重文件加载参数，并更新 model 中的模型权重
        # model：需要加载权重的 PyTorch 模型实例
        # weight_path：预训练权重文件路径
        safetensors.torch.load_model(model, weight_path)

        # 返回包含预训练权重的模型实例
        return model

    # @abc.abstractmethod：声明抽象方法，要求具体子类实现
    @abc.abstractmethod
    # inputs_spec（Input Specification）：返回模型输入的抽象数据规格
    # Observation 和 Actions 中的张量均由 jax.ShapeDtypeStruct 描述，仅包含 shape 和 dtype，没有真实数据
    def inputs_spec(
        self,
        *,
        # batch_size：批次大小（Batch Size），默认 1
        batch_size: int = 1
    ) -> tuple[Observation, Actions]: ...

    # fake_obs（Fake Observation）：根据 Observation 的抽象数据规格，生成所有张量元素均为 1 的模拟观测数据
    def fake_obs(
        self,
        # batch_size：批次大小（Batch Size），默认 1
        batch_size: int = 1
    ) -> Observation:

        # inputs_spec：获取模型输入的抽象数据规格：Observation 和 Actions
        # observation_spec：观测数据的抽象规格，包含 shape 和 dtype
        # _：忽略 Actions 的抽象数据规格
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)

        # jax.tree.map：遍历 observation_spec 的所有叶子节点
        # x：当前叶子节点的抽象规格 jax.ShapeDtypeStruct，包含 shape 和 dtype
        # jnp.ones：根据 x.shape 和 x.dtype 创建元素全为 1 的真实张量
        # 返回：与 observation_spec 结构相同、包含真实数据的 Observation
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)


    # fake_act（Fake Action）：根据 Actions 的抽象数据规格，生成所有张量元素均为 1 的模拟动作数据
    def fake_act(
        self,
        # batch_size：批次大小（Batch Size），默认 1
        batch_size: int = 1
    ) -> Actions:

        # inputs_spec：获取模型输入的抽象数据规格：Observation 和 Actions
        # action_spec：动作数据的抽象规格，包含 shape 和 dtype
        # _：忽略 Observation 的抽象数据规格
        _, action_spec = self.inputs_spec(batch_size=batch_size)

        # jax.tree.map：遍历 action_spec 的所有叶子节点
        # x：当前叶子节点的抽象规格 jax.ShapeDtypeStruct，包含 shape 和 dtype
        # jnp.ones：根据 x.shape 和 x.dtype 创建元素全为 1 的真实张量
        # 返回：与 action_spec 结构相同、包含真实数据的 Actions
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

# Checkpoint（检查点）：训练过程中保存的模型参数及其他训练状态，用于恢复训练或模型推理
# restore_params（Restore Parameters）：从 Checkpoint 中读取并恢复模型参数
def restore_params(
    # params_path：Checkpoint 所在目录的路径，支持 Path 对象或字符串
    params_path: pathlib.Path | str,
    *,
    # restore_type：恢复后的参数数组类型（Array Type），默认 jax.Array，也可指定 np.ndarray
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    # dtype：恢复后的参数数据类型（Data Type），默认 None，保留 Checkpoint 中原始 dtype
    dtype: jnp.dtype | None = None,
    # sharding：参数的设备分片策略（Sharding），默认 None，使用函数内部的默认策略
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:

    # 处理 Checkpoint 路径：本地路径转换为绝对路径，GCS 云存储路径保持不变
    # 1. params_path 不以 "gs://" 开头（本地路径）：
    #    1. pathlib.Path(params_path)：将本地路径转换为 Path 路径对象
    #    2. resolve()：将路径转换为绝对路径，同时解析符号链接
    # 2. params_path 以 "gs://" 开头（Google Cloud Storage 云存储路径）：
    #    保持原路径不变
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    # 如果恢复后的参数数组类型为 jax.Array，且未指定设备分片策略，则创建默认的设备分片策略
    if restore_type is jax.Array and sharding is None:

        # 1. jax.devices()：获取当前 JAX 可以使用的所有计算设备
        # 2. ("x",)：将设备网格的唯一维度命名为 "x"
        # 3. jax.sharding.Mesh：将所有设备组织为一维设备网格
        # 4. mesh：一维设备网格，shape=[设备数量]
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))

        # 1. mesh：设备网格（Device Mesh），指定参与参数存储的计算设备
        # 2. jax.sharding.PartitionSpec()：空分片规则，表示参数不沿设备网格维度进行切分
        # 3. jax.sharding.NamedSharding：将设备网格与分片规则关联，创建分片策略
        # 4. sharding：参数在 mesh 的所有设备上复制，每个设备保存完整的参数数据
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # ocp.PyTreeCheckpointer：创建 Orbax Checkpoint 读取器，用于恢复 PyTree 结构的模型参数
    # ckptr：Checkpoint 读取器
    # with：使用结束后自动释放 Checkpointer 管理的资源
    with ocp.PyTreeCheckpointer() as ckptr:

        # ckptr.metadata：读取 Checkpoint 的元数据（Metadata），不加载真实权重数值
        # params_path：Checkpoint 所在目录
        # metadata：Checkpoint 的元数据，包含参数结构、shape、dtype 等信息
        metadata = ckptr.metadata(params_path)

        # metadata["params"]：获取模型参数对应的元数据 PyTree
        # item：构建只包含 "params" 的字典，指定需要恢复的参数部分
        item = {"params": metadata["params"]}

        # params：包含真实权重数值的模型参数 PyTree 
        # ckptr.restore：从 Checkpoint 中恢复真实模型参数
        params = ckptr.restore(
            # params_path：Checkpoint 所在目录
            params_path,
            # ocp.args.PyTreeRestore：配置 PyTree 的恢复方式
            ocp.args.PyTreeRestore(
                # item：指定需要恢复的参数结构
                item=item,
                # restore_args：为 PyTree 中的每个参数指定恢复配置
                # jax.tree.map：遍历 item 中的所有叶子节点
                # _：当前叶子节点的元数据，此处不使用其具体值
                # ocp.ArrayRestoreArgs：为每个参数创建数组恢复配置
                # sharding：指定参数恢复后的设备分片策略
                # restore_type：指定参数恢复后的数组类型，例如 jax.Array
                # dtype：指定参数恢复后的数据类型，None 表示保留原始 dtype
                # 返回：与 item 具有相同 PyTree 结构的恢复配置
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"] # ["params"]：从恢复结果中提取模型参数


    # traverse_util.flatten_dict：将嵌套的参数字典展平为以元组路径为键的字典
    # params：从 Checkpoint 中恢复的真实模型参数 PyTree
    # flat_params：展平后的参数字典
    #
    # 例如：
    # params = {
    #     "layer1": {
    #         "kernel": {"value": 10},
    #         "bias": {"value": 20}
    #     }
    # }
    #
    # flat_params = {
    #     ("layer1", "kernel", "value"): 10,
    #     ("layer1", "bias", "value"): 20
    # }
    flat_params = traverse_util.flatten_dict(params)


    if all(kp[-1] == "value" for kp in flat_params):
        # flat_params.items()：遍历参数字典中的键路径 kp 和对应的参数值 v
        # kp[:-1]：移除键路径最后一个元素 "value"
        # v：对应的真实模型参数数组，保持不变
        # flat_params：移除所有参数路径末尾 "value" 后的新字典
        #
        # flat_params = {
        #     ("layer1", "kernel"): 10,
        #     ("layer1", "bias"): 20
        # }
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}

    # traverse_util.unflatten_dict：将展平的参数字典重新转换为嵌套字典
    # 返回：包含真实权重数值的纯参数字典（Pure Dict）
    #
    # {
    #     "layer1": {
    #         "kernel": 10,
    #         "bias": 20
    #     }
    # }
    return traverse_util.unflatten_dict(flat_params)
