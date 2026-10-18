model: Pi0
    action_dim = 32                              # 动作维度
    action_horizon = 50                          # 连续预测动作数
    max_token_len = 48                           # 最长语言 Token 数
    pi05 = False                                 # 不使用 Pi0.5 模型

    paligemma_config: Config                     # VLM 模型 gemma_2b 的配置参数
        width = 2048                             # 隐藏层维度
        depth = 18                               # Transformer Block 层数
        mlp_dim = 16384                          # MLP 层维度
        num_heads = 8                            # Query 头数
        num_kv_heads = 1                         # Key 和 Value 头数
        head_dim = 256                           # 每个头的维数
        lora_configs = {}                        # LoRA 微调参数

    action_expert_config: Config                 # AE 模型 gemma_300m 的配置参数
        width = 1024                             # 隐藏层维度
        depth = 18                               # Transformer Block 层数
        mlp_dim = 4096                           # MLP 层维度
        num_heads = 8                            # Query 头数
        num_kv_heads = 1                         # Key 和 Value 头数
        head_dim = 256                           # 每个头的维数
        lora_configs = {}                        # LoRA 微调参数

    llm: nnx_bridge.ToNNX                        # Linen 与 NNX 的桥接模块
        module: gemma.Module                     # VLM 和 AE 模型汇总
            configs = [paligemma_config, action_expert_config]  # VLM 和 AE 的配置参数
            embed_dtype = "bfloat16"             # 词嵌入数据类型
            dropout = 0.0                        # 每个位置被丢弃的概率
            dropout_bdims = ()                   # 指定哪些维度共享 Dropout 掩码
            adarms = False                       # 不使用 Adaptive RMSNorm

            embedder: Embedder                   # 词嵌入器
            	vocab_size = 257152              # 词表大小
                embed_dim = 2048                 # 词嵌入维度
		        input_embedding_table # [vocab_size(257152), embed_dim(2048)]		

            layers: nn.scan                      # 多层 Transformer Block
                length = 18                      # Transformer Block 层数
                variable_axes = {"params": 0}    # 可训练权重沿第 0 维堆叠
                split_rngs = {                   # 各层随机密钥配置
                    "params": True,              # 每层使用不同的参数初始化随机密钥
                    "dropout": True              # 每层使用不同的 Dropout 随机密钥
                }
                in_axes = (                      
							# xs 为 carry，自动传递至下一层
                    0,                           # kv_cache 沿第 0 维拆分
                    nn.broadcast,                # positions 所有 Block 共享
                    nn.broadcast,                # mask 所有 Block 共享
                    nn.broadcast,                # adarms_cond 所有 Block 共享
                    nn.broadcast                 # deterministic 所有 Block 共享
                )
                block_cls: nn.remat(Block)       # 使用重计算机制包装 Block
                block: Block                    # Transformer Block
                    configs = [paligemma_config, action_expert_config]  # VLM 和 AE 的配置参数
                    dropout = 0.0                # Dropout 丢弃概率
                    dropout_bdims = ()           # Dropout 掩码广播维度

            final_norms = [                      # 最终归一化模块列表
                final_norm: RMSNorm              # VLM 最终归一化模块
                final_norm_1: RMSNorm            # AE 最终归一化模块
            ]

    img: nnx_bridge.ToNNX                        # Linen 与 NNX 的桥接模块
    	module: siglip.Module                     # 视觉编码器
            num_classes = 2048             # 分类数 = paligemma_config 中的隐藏层维数
            patch_size = (14, 14)
            width = 1152
            depth = 27
            mlp_dim = 4304
            num_heads = 16
            posemb = "learn"
            rep_size = False
            dropout = 0.0
            pool_type = "none"
            head_zeroinit = True
            scan = True
            remat_policy = "nothing_saveable"
            dtype_mm = "bfloat16"
            
    PaliGemma: nnx.Dict                         # 组合 SigLIP 视觉编码器和 Gemma Transformer（包含 VLM 与 Action Expert）
        llm: llm                                # Gemma Transformer 模块
        img: img                                # SigLIP 视觉编码器

    action_in_proj: nnx.Linear                  # 动作输入投影层，将动作映射到 Action Expert 的隐藏特征空间
        in_features = 32                        # 输入维度：action_dim
        out_features = 1024                     # 输出维度：action_expert_config.width

    state_proj: nnx.Linear                      # 机器人状态投影层，将状态映射到 Action Expert 的隐藏特征空间
        in_features = 32                        # 输入维度：action_dim
        out_features = 1024                     # 输出维度：action_expert_config.width

    action_time_mlp_in: nnx.Linear              # 动作与时间嵌入融合后的 MLP 输入层
        in_features = 2048                      # 输入维度：2 * action_expert_config.width
        out_features = 1024                     # 输出维度：action_expert_config.width

    action_time_mlp_out: nnx.Linear             # 动作与时间嵌入融合后的 MLP 输出层
        in_features = 1024                      # 输入维度：action_expert_config.width
        out_features = 1024                     # 输出维度：action_expert_config.width

    action_out_proj: nnx.Linear                 # 动作输出投影层，将 Action Expert 的特征映射回动作空间
        in_features = 1024                      # 输入维度：action_expert_config.width
        out_features = 32                       # 输出维度：action_dim

    deterministic = True                        # 确定性模式：不进行 Dropout 随机丢弃
