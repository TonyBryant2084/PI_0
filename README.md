# OpenPI — 中文 README

OpenPI 包含由 Physical Intelligence 发布的开源机器人模型和软件包。

目前，该仓库包含三种类型的模型：

- [π₀ 模型](https://www.physicalintelligence.company/blog/pi0?utm_source=chatgpt.com)：一种基于流（Flow-based）的视觉-语言-动作模型（Vision-Language-Action，VLA）。
- [π₀-FAST 模型](https://www.physicalintelligence.company/research/fast?utm_source=chatgpt.com)：一种基于 FAST 动作分词器（Action Tokenizer）的自回归 VLA 模型。
- [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05?utm_source=chatgpt.com)：π₀ 的升级版本，通过[知识隔离（Knowledge Insulation）](https://www.physicalintelligence.company/research/knowledge_insulation?utm_source=chatgpt.com)训练，具有更好的开放世界泛化能力。请注意，目前本仓库仅支持使用 Flow Matching 动作头对 π₀.₅ 进行训练和推理。

对于所有模型，我们都提供了在超过 10,000 小时机器人数据上预训练的基础模型（Base Model）检查点，以及直接使用这些模型或在自定义数据集上进行微调的示例。

这是一个实验性项目：π₀ 最初是针对我们自己的机器人开发的，这些机器人与广泛使用的 [ALOHA](https://tonyzhaozh.github.io/aloha/?utm_source=chatgpt.com) 和 [DROID](https://droid-dataset.github.io/?utm_source=chatgpt.com) 等平台存在差异。

虽然我们乐观地认为，研究人员和开发者能够通过创造性实验将 π₀ 适配到自己的机器人平台，但我们并不认为每一次尝试都会成功。

换句话说，π₀ 可能适用于你的机器人，也可能不适用，但我们欢迎你亲自尝试并验证效果！

## 更新记录（Updates）

- 2025 年 9 月：OpenPI 正式增加了 PyTorch 支持。
- 2025 年 9 月：发布 π₀.₅，它是 π₀ 的升级版本，具有更好的开放世界泛化能力。
- 2025 年 9 月：为 DROID 训练增加了[改进的空闲数据过滤器](https://examples/droid/README_train.md?utm_source=chatgpt.com#data-filtering)。
- 2025 年 6 月：增加了使用 `openpi` 在完整 [DROID 数据集](https://droid-dataset.github.io/?utm_source=chatgpt.com)上训练 VLA 模型的[说明文档](https://examples/droid/README_train.md?utm_source=chatgpt.com)。这是用于训练 π₀-FAST-DROID 的训练流程的一个近似开源实现。

## 环境要求（Requirements）

要运行本仓库中的模型，你需要一块满足以下最低规格要求的 NVIDIA GPU。

以下显存估算基于单 GPU 环境。不过，你也可以通过在训练配置中设置 `fsdp_devices`，利用多 GPU 模型并行来降低每块 GPU 的显存需求。

另外，请注意，目前的训练脚本尚不支持多节点训练。

| 运行模式                    | 所需显存      | 示例 GPU           |
| ----------------------- | --------- | ---------------- |
| 推理（Inference）           | > 8 GB    | RTX 4090         |
| LoRA 微调（Fine-Tuning）    | > 22.5 GB | RTX 4090         |
| 全参数微调（Full Fine-Tuning） | > 70 GB   | A100 80GB / H100 |

本仓库已经在 Ubuntu 22.04 上进行测试，目前尚未正式支持其他操作系统。

## 安装（Installation）

克隆本仓库时，请确保同时更新 Git 子模块：

```
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# 如果你已经克隆了仓库，则执行：
git submodule update --init --recursive
```

我们使用 [uv](https://docs.astral.sh/uv/?utm_source=chatgpt.com) 管理 Python 依赖。

请参考 [uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/?utm_source=chatgpt.com)完成安装。

安装 uv 后，运行以下命令配置环境：

```
GIT_LFS_SKIP_SMUDGE=1 uv sync

GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

注意： 需要设置 `GIT_LFS_SKIP_SMUDGE=1`，才能正确拉取作为依赖项的 LeRobot。

Docker： 除了使用 uv 安装，我们还提供了通过 Docker 安装 OpenPI 的说明。

如果在配置系统环境时遇到问题，可以考虑使用 Docker 简化安装过程。

更多信息请参阅 [Docker 安装文档](https://docs/docker.md?utm_source=chatgpt.com)。

## 模型检查点（Model Checkpoints）

### 基础模型（Base Models）

我们提供了多个基础 VLA 模型检查点。

这些检查点已经在超过 10,000 小时的机器人数据上完成预训练，可以用于后续微调。

| 模型      | 用途 | 说明                                                                                                     | 检查点路径                                          |
| ------- | -- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------- |
| π₀      | 微调 | 用于微调的基础 [π₀ 模型](https://www.physicalintelligence.company/blog/pi0?utm_source=chatgpt.com)              | `gs://openpi-assets/checkpoints/pi0_base`      |
| π₀-FAST | 微调 | 用于微调的基础自回归 [π₀-FAST 模型](https://www.physicalintelligence.company/research/fast?utm_source=chatgpt.com) | `gs://openpi-assets/checkpoints/pi0_fast_base` |
| π₀.₅    | 微调 | 用于微调的基础 [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05?utm_source=chatgpt.com)           | `gs://openpi-assets/checkpoints/pi05_base`     |

### 微调模型（Fine-Tuned Models）

我们还提供了适用于不同机器人平台和任务的“专家”模型检查点。

这些模型是在上述基础模型的基础上微调得到的，目的是直接在目标机器人上运行。

不过，它们不一定能够在你的特定机器人上正常工作。

由于这些检查点是在 ALOHA、DROID Franka 等较为常见的机器人平台上，使用相对较小的数据集进行微调的，因此不一定能够泛化到你的具体机器人配置。

尽管如此，我们在实际实验中发现，部分模型（尤其是 DROID 检查点）具有相当广泛的泛化能力。

| 模型                  | 用途      | 说明                                                                                                                                                                                     | 检查点路径                                                 |
| ------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| π₀-FAST-DROID       | 推理      | 在 [DROID 数据集](https://droid-dataset.github.io/?utm_source=chatgpt.com)上微调的 π₀-FAST 模型，可在 DROID 机器人平台的新场景中以零样本方式完成多种简单桌面操作任务                                                            | `gs://openpi-assets/checkpoints/pi0_fast_droid`       |
| π₀-DROID            | 微调      | 在 DROID 数据集上微调的 π₀ 模型，推理速度比 π₀-FAST-DROID 更快，但对语言指令的遵循能力可能较弱                                                                                                                           | `gs://openpi-assets/checkpoints/pi0_droid`            |
| π₀-ALOHA-towel      | 推理      | 使用内部 ALOHA 数据微调的 π₀ 模型，可在 ALOHA 机器人平台上零样本折叠多种毛巾                                                                                                                                        | `gs://openpi-assets/checkpoints/pi0_aloha_towel`      |
| π₀-ALOHA-tupperware | 推理      | 使用内部 ALOHA 数据微调的 π₀ 模型，可从塑料保鲜盒中取出食物                                                                                                                                                    | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| π₀-ALOHA-pen-uncap  | 推理      | 使用公开 [ALOHA 数据](https://dit-policy.github.io/?utm_source=chatgpt.com)微调的 π₀ 模型，可完成拔除笔帽任务                                                                                               | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |
| π₀.₅-LIBERO         | 推理      | 针对 [LIBERO 基准测试](https://libero-project.github.io/datasets?utm_source=chatgpt.com)微调的 π₀.₅ 模型，取得了当时最先进的性能，详见 [LIBERO README](https://examples/libero/README.md?utm_source=chatgpt.com) | `gs://openpi-assets/checkpoints/pi05_libero`          |
| π₀.₅-DROID          | 推理 / 微调 | 使用[知识隔离](https://www.physicalintelligence.company/research/knowledge_insulation?utm_source=chatgpt.com)方法在 DROID 数据集上微调的 π₀.₅ 模型，具有较快的推理速度和良好的语言指令遵循能力                                 | `gs://openpi-assets/checkpoints/pi05_droid`           |

默认情况下，模型检查点会根据需要自动从 `gs://openpi-assets` 下载，并缓存在：

```
~/.cache/openpi
```

你可以通过设置环境变量 `OPENPI_DATA_HOME` 修改下载目录。

## 运行预训练模型推理（Running Inference for a Pre-Trained Model）

只需要几行代码，就可以运行我们提供的预训练模型检查点。

下面是一个推理示例。原文将其描述为 π₀-FAST-DROID 示例，但实际代码使用的是 `pi05_droid` 配置和权重。

```
from openpi.training import config as _configfrom openpi.policies import policy_configfrom openpi.shared import downloadconfig = _config.get_config("pi05_droid")checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")# 创建一个已训练的策略模型policy = policy_config.create_trained_policy(config, checkpoint_dir)# 使用一个示例观测运行推理example = {    "observation/exterior_image_1_left": ...,    "observation/wrist_image_left": ...,    ...    "prompt": "pick up the fork"}action_chunk = policy.infer(example)["actions"]
```

你也可以通过[示例 Notebook](https://examples/inference.ipynb?utm_source=chatgpt.com) 测试上述过程。

我们还提供了详细的分步示例，介绍如何在以下机器人平台上运行预训练模型推理：

- [DROID 机器人](https://examples/droid/README.md?utm_source=chatgpt.com)
- [ALOHA 机器人](https://examples/aloha_real/README.md?utm_source=chatgpt.com)

远程推理（Remote Inference）：

我们提供了通过远程方式运行模型推理的[示例和代码](https://docs/remote_inference.md?utm_source=chatgpt.com)。

模型可以运行在另一台服务器上，并通过 WebSocket 连接将动作持续传输给机器人。

这样，你就可以在机器人之外的服务器上使用性能更强的 GPU，同时将机器人运行环境与策略模型运行环境分离。

无需机器人即可测试推理：

我们还提供了一个[测试脚本](https://examples/simple_client/README.md?utm_source=chatgpt.com)，可以在没有真实机器人的情况下测试模型推理。

该脚本会生成随机观测数据，并使用模型执行推理。

更多信息请参阅[简单客户端说明](https://examples/simple_client/README.md?utm_source=chatgpt.com)。

## 在自定义数据上微调基础模型（Fine-Tuning Base Models on Your Own Data）

我们将以在 [LIBERO 数据集](https://libero-project.github.io/datasets?utm_source=chatgpt.com)上微调 π₀.₅ 模型为例，介绍如何使用自己的数据微调基础模型。

整个过程分为三个步骤：

1. 将数据转换为 LeRobot 数据集格式（我们使用该格式进行训练）。
2. 定义训练配置并运行训练。
3. 启动策略服务器并执行推理。

### 1. 将数据转换为 LeRobot 数据集

我们在 [`examples/libero/convert_libero_data_to_lerobot.py`](https://examples/libero/convert_libero_data_to_lerobot.py?utm_source=chatgpt.com) 中提供了一个最小化示例脚本，用于将 LIBERO 数据转换为 LeRobot 数据集。

你可以很方便地修改该脚本，将自己的数据转换为相同格式。

可以从[这里](https://huggingface.co/datasets/openvla/modified_libero_rlds?utm_source=chatgpt.com)下载原始 LIBERO 数据集，然后运行：

```
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

注意： 如果你只想在 LIBERO 数据集上进行微调，可以跳过这一步，因为我们提供的 LIBERO 微调配置已经指向预先转换好的 LIBERO 数据集。

这里的数据转换步骤主要是一个示例，你可以将其修改后应用于自己的数据。

### 2. 定义训练配置并运行训练

要使用自己的数据微调基础模型，需要定义数据处理配置和训练配置。

下面提供了带有详细注释的 LIBERO 示例配置，你可以根据自己的数据集进行修改：

- [`LiberoInputs`](https://src/openpi/policies/libero_policy.py?utm_source=chatgpt.com)[ 和 ](https://src/openpi/policies/libero_policy.py?utm_source=chatgpt.com)[`LiberoOutputs`](https://src/openpi/policies/libero_policy.py?utm_source=chatgpt.com)：定义 LIBERO 环境与模型之间的双向数据映射。这些映射会同时用于训练和推理。
- [`LeRobotLiberoDataConfig`](https://src/openpi/training/config.py?utm_source=chatgpt.com)：定义如何处理 LeRobot 数据集中的原始 LIBERO 数据，以供模型训练。
- [`TrainConfig`](https://src/openpi/training/config.py?utm_source=chatgpt.com)：定义微调超参数、数据配置以及权重加载器。

我们还提供了在 LIBERO 数据上微调 [π₀](https://src/openpi/training/config.py?utm_source=chatgpt.com)、[π₀-FAST](https://src/openpi/training/config.py?utm_source=chatgpt.com) 和 [π₀.₅](https://src/openpi/training/config.py?utm_source=chatgpt.com) 的示例配置。

在开始训练之前，需要计算训练数据的归一化统计量。

运行以下脚本，并指定训练配置名称：

```
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

接下来，就可以通过以下命令启动训练：

```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

其中，`--overwrite` 参数用于在使用相同配置重新执行微调时，覆盖已经存在的模型检查点。

该命令会将训练进度输出到终端，并将模型检查点保存到 `checkpoints` 目录。

你也可以通过 Weights & Biases 控制面板监控训练进度。

为了最大限度地利用 GPU 显存，可以在启动训练前设置：

```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

这样，JAX 最多可以使用 90% 的 GPU 显存，而默认情况下为 75%。

注意： 我们提供了重新加载预训练阶段状态和动作归一化统计量的功能。

如果你正在为预训练数据中已经出现过的机器人微调新任务，那么重新加载这些统计量可能会带来帮助。

有关重新加载归一化统计量的详细说明，请参阅 [`norm_stats.md`](https://docs/norm_stats.md?utm_source=chatgpt.com)。

### 3. 启动策略服务器并运行推理

训练完成后，可以启动策略服务器，再通过 LIBERO 评估脚本向服务器发送请求，执行模型推理。

启动模型服务器非常简单。

下面以训练到第 20,000 步时保存的检查点为例，你可以根据需要修改路径：

```
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

这条命令会启动一个监听 `8000` 端口的服务器，等待客户端向其发送观测数据。

随后，可以运行评估脚本（或机器人运行程序），向该服务器发送推理请求。

对于 LIBERO 评估，我们提供并推荐使用基于 Docker 的工作流程，它可以同时管理策略服务器和评估脚本。

更多信息请参阅 [LIBERO README](https://examples/libero/README.md?utm_source=chatgpt.com)。

如果你希望在自己的机器人运行程序中调用策略服务器，可以参考[远程推理文档](https://docs/remote_inference.md?utm_source=chatgpt.com)中的最小化示例。

### 更多示例（More Examples）

我们还在以下 README 文档中提供了更多示例，介绍如何在 ALOHA 平台及其他机器人平台上微调模型并执行推理：

- ALOHA 仿真器
- ALOHA 真实机器人
- UR5 机器人

## PyTorch 支持（PyTorch Support）

现在，OpenPI 除了原有的 JAX 实现之外，还提供了 π₀ 和 π₀.₅ 模型的 PyTorch 实现！

PyTorch 实现已经在 LIBERO 基准测试上通过验证，包括推理和微调。

目前仍有部分功能尚未得到支持（未来可能发生变化）：

- π₀-FAST 模型。
- 混合精度训练（Mixed Precision Training）。
- FSDP（Fully Sharded Data Parallel，完全分片数据并行）训练。
- LoRA（Low-Rank Adaptation，低秩适配）训练。
- 训练期间使用 EMA（Exponential Moving Average，指数移动平均）权重。

### 环境配置（Setup）

1\. 确保已经安装所有依赖项的最新版本：

```
uv sync
```

2\. 再次确认已经安装 transformers 4.53.2：

```
uv pip show transformers
```

3\. 应用 transformers 库补丁：

```
cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
```

该命令会覆盖 transformers 库中的多个文件，以实现必要的模型修改：

1. 支持 AdaRMS。
2. 正确控制激活值的计算精度。
3. 允许使用 KV Cache，但不更新缓存内容。

警告： 使用 uv 默认的链接模式（硬链接）时，这一操作会永久影响 uv 缓存中的 transformers 库。

这意味着，即使重新安装 transformers，这些修改仍然可能保留，甚至可能影响其他使用 transformers 的项目。

要完全撤销该操作，必须运行：

```
uv cache clean transformers
```

### 将 JAX 模型转换为 PyTorch（Converting JAX Models to PyTorch）

要将 JAX 模型检查点转换为 PyTorch 格式，可以执行：

```
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### 使用 PyTorch 运行推理（Running Inference with PyTorch）

PyTorch 实现使用与 JAX 版本相同的 API。

你只需要修改模型检查点路径，使其指向转换后的 PyTorch 模型即可：

```
from openpi.training import config as _configfrom openpi.policies import policy_configfrom openpi.shared import downloadconfig = _config.get_config("pi05_droid")checkpoint_dir = "/path/to/converted/pytorch/checkpoint"# 创建已训练的策略模型（自动检测 PyTorch 格式）policy = policy_config.create_trained_policy(config, checkpoint_dir)# 执行推理（与 JAX 使用相同的 API）action_chunk = policy.infer(example)["actions"]
```

### 使用 PyTorch 启动策略服务器（Policy Server with PyTorch）

策略服务器对 PyTorch 模型的使用方式与 JAX 模型完全相同。

只需要将模型路径指向转换后的 PyTorch 检查点目录：

```
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### 使用 PyTorch 微调（Finetuning with PyTorch）

要在 PyTorch 中微调模型，请按照以下步骤操作。

1\. 将 JAX 基础模型转换为 PyTorch 格式：

```
uv run examples/convert_jax_model_to_pytorch.py \
    --config_name <config name> \
    --checkpoint_dir /path/to/jax/base/model \
    --output_path /path/to/pytorch/base/model
```

2\. 在训练配置中使用 `pytorch_weight_path` 指定转换后的 PyTorch 模型路径。

3\. 使用以下任一种模式启动训练。

单 GPU 训练：

```
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
```

示例：

```
uv run scripts/train_pytorch.py debug --exp_name pytorch_test

# 从最新检查点恢复训练
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume
```

多 GPU 训练（单节点）：

```
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
```

示例：

```
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test

uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
```

多节点训练：

```
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### 精度设置（Precision Settings）

JAX 和 PyTorch 实现对计算精度的处理方式如下。

JAX：

1. 推理： 大部分权重和计算使用 `bfloat16`，少量计算使用 `float32` 以保证数值稳定性。
2. 训练： 默认使用混合精度训练。权重和梯度使用 `float32`，大部分激活值和计算使用 `bfloat16`。你可以通过在配置中将 `dtype` 设置为 `float32`，切换到完整的 `float32` 训练。

PyTorch：

1. 推理： 与 JAX 保持一致，大部分权重和计算使用 `bfloat16`，少量权重转换为 `float32` 以保证数值稳定性。
2. 训练： 支持完整 `bfloat16`（默认）或完整 `float32` 训练。可以通过配置中的 `pytorch_training_precision` 修改精度。`bfloat16` 使用的显存更少，但与 `float32` 相比，训练损失通常更高。目前尚不支持混合精度训练。

使用 `torch.compile` 后，JAX 和 PyTorch 的推理速度基本相当。

## 故障排查（Troubleshooting）

我们会在这里收集常见问题及其解决方案。

如果遇到问题，请首先查看本章节。如果仍然无法找到解决方案，请在仓库中提交 Issue，并参考[贡献指南](https://CONTRIBUTING.md?utm_source=chatgpt.com)。

| 问题                 | 解决方案                                                                                                                                                                                                                                                                     |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `uv sync` 因依赖冲突而失败 | 尝试删除虚拟环境目录（`rm -rf .venv`），然后重新运行 `uv sync`。如果问题仍然存在，请检查是否已经安装最新版本的 uv（`uv self update`）。                                                                                                                                                                                |
| 训练时 GPU 显存不足       | 确保在训练前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`（或更高），允许 JAX 使用更多 GPU 显存。还可以使用 `--fsdp-devices <n>` 启用[完全分片数据并行](https://engineering.fb.com/2021/07/15/open-source/fsdp/?utm_source=chatgpt.com)，其中 `<n>` 是 GPU 数量。该方法能够降低显存占用，但会使训练速度下降，具体下降程度取决于硬件配置。如果仍然显存不足，可以考虑禁用 EMA。 |
| 策略服务器连接错误          | 检查服务器是否正在运行，以及是否监听预期端口。确认客户端与服务器之间的网络连接和防火墙配置正确。                                                                                                                                                                                                                         |
| 训练时提示缺少归一化统计量      | 在开始训练之前，使用对应的训练配置名称运行 `scripts/compute_norm_stats.py`。                                                                                                                                                                                                                   |
| 数据集下载失败            | 检查网络连接。对于 Hugging Face 数据集，确保已经登录（`huggingface-cli login`）。                                                                                                                                                                                                              |
| CUDA/GPU 错误        | 确认 NVIDIA 驱动程序已经正确安装。如果使用 Docker，请确保安装了 `nvidia-container-toolkit`。检查 GPU 兼容性。你不需要在系统层面单独安装 CUDA 库，因为 uv 会自动安装所需的 CUDA 依赖。如果遇到 CUDA 问题，甚至可以尝试卸载系统层面的 CUDA 库，因为它们有时可能造成依赖冲突。                                                                                              |
| 运行示例时出现导入错误        | 确保已经通过 `uv sync` 安装所有依赖。部分示例可能还有额外的依赖要求，具体请查看相应的 README。                                                                                                                                                                                                                 |
| 动作维度不匹配            | 检查数据处理变换是否与机器人预期的输入和输出维度一致。检查策略类中的动作空间定义。                                                                                                                                                                                                                                |
| 训练损失发散             | 检查数据集 `norm_stats.json` 中的 `q01`、`q99` 和 `std` 数值。某些很少使用的维度可能具有非常小的 `q01`、`q99` 或 `std`，导致归一化后的状态和动作数值异常巨大。可以通过手动调整归一化统计量暂时解决。                                                                                                                                           |

以上为你上传的 OpenPI README 全部章节的中文翻译，保留了原文的技术术语、模型检查点路径、命令、配置参数以及相关链接。