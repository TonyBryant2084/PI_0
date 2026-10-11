import jax
import jax.numpy as jnp

from openpi.models.pi0_config import Pi0Config

def main():

    # Jax 当前版本 (JAX version: 0.5.3)
    print("JAX version:", jax.__version__)

    # 识别运行设备 (Devices: [CudaDevice(id=0)])
    print("Devices:", jax.devices())

    # 2. 配置 Pi0 模型
    config = Pi0Config(
        dtype="bfloat16",
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        action_dim=32,
        action_horizon=50,
    )

    # 3. 随机初始化模型，不加载预训练权重
    rng = jax.random.PRNGKey(0)
    model = config.create(rng)

    print("Model initialized.")

    # 4. 创建假观测数据
    observation = config.fake_obs(batch_size=1)

    # 5. 创建随机动作序列
    actions = jax.random.normal(
        jax.random.PRNGKey(1),
        shape=(1, config.action_horizon, config.action_dim),
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
