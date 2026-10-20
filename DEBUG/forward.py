import jax
import jax.numpy as jnp

from openpi.models.pi0_config import Pi0Config

def main():

    # Jax 当前版本 (JAX version: 0.5.3)
    print("JAX version:", jax.__version__)

    # 获取 JAX 当前可用的所有设备，例如 CPU、GPU (Devices: [CudaDevice(id=0)])
    print("Devices:", jax.devices())

# 2. 配置 Pi0 模型
    # config = Pi0Config(
    #     dtype="bfloat16",                     # 词嵌入结果的指定类型，模型计算使用 bfloat16 精度
    #     paligemma_variant="gemma_2b",         # 使用 gemma_2b 配置的 PaliGemma 视觉语言模型
    #     action_expert_variant="gemma_300m",   # 使用 gemma_300m 配置的 Action Expert 动作模型
    #     action_dim=32,                        # 每个动作的维度为 32
    #     action_horizon=50,                    # 每次预测连续 50 个动作
    #     max_token_len=48,                     # 语言提示最多包含 48 个 token
    #     discrete_state_input=False,           # 不将状态输入作为离散 token 处理
    #     pi05=False,                           # 不使用 Pi0.5 模型配置
    #     pytorch_compile_mode="max-autotune",  # 使用 PyTorch 编译的 max-autotune 模式，最大化自动调优
    # )
    config = Pi0Config(
        dtype="bfloat16",
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=32,
        action_horizon=50,
    )

    # 3. 随机初始化模型，不加载预训练权重
    #   1) 创建 JAX 随机数密钥，用于控制随机数的生成，使用整数 0 作为随机种子，便于复现实验    
    rng = jax.random.PRNGKey(0)
    #   2) 创建 PI0 模型
    model = config.create(rng)

    print("Model initialized.")

    # 4. 创建假观测数据
    observation = config.fake_obs(batch_size=5)

    # 5. 创建随机动作序列
    actions = jax.random.normal(
        jax.random.PRNGKey(1),
        shape=(5, config.action_horizon, config.action_dim),
        dtype=jnp.float32,
    )

    print("Observation state shape:", observation.state.shape)
    print("Action shape:", actions.shape)

    # 6. 计算 Pi0 的流匹配损失
    loss_rng = jax.random.PRNGKey(2)
    loss = model.compute_loss(
        loss_rng,
        observation,
        actions,
        train=False,
    )

    # 7. 等待 GPU 计算完成，再读取结果
    loss.block_until_ready()

    print("Loss shape:", loss.shape)
    print("Loss:", loss)


if __name__ == "__main__":
    main()
