import torch
import torch.nn as nn
import src.part04.ch0101_dummy_gpt_model as p4c1
import src.part04.ch0201_layer_normalization as p4c2
import src.part04.ch0301_feed_forward as p4c3
import src.part04.ch0401_deep_neural_network as p4c4
import src.part04.ch0501_transformer_block as p4c5
import tiktoken

def test_ch0101():
    """
        vocab_size指的是第 2 章中 BPE 分词器使用的 50,257 个词汇的词表大小
        context_length表示模型所能处理的最大输入 token 数（在第 2 章介绍位置嵌入时讨论过）​
        emb_dim表示嵌入维度，将每个 token 转换为 768 维的向量
        n_layers指定模型中 Transformer 模块的层数，后续章节将对此详解
        drop_rate表示 dropout 机制的强度（例如，0.1 表示丢弃 10% 的隐藏单元）​，用于防止过拟合，具体内容请回顾第 3 章
        qkv_bias 参数决定是否在多头注意力的查询、键和值的线性层中加入偏置向量。我们最初会禁用该选项，以遵循现代大语言模型的标准，之后在第 6 章加载 OpenAI 预训练的 GPT-2 权重时再重新考虑该设置。
    """
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-Key-Value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    # 因为txt1/2是分词后的长度恰好一致，所以得到到张量是一致的，可以合并为一个批次
    # 将多个独立的token序列合并成一个批次(batch
    batch = torch.stack(batch, dim=0)
    print(batch)

    torch.manual_seed(123)
    model = p4c1.DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)


def test_ch0201():
    # 设置随机种子，保证结果可复现
    torch.manual_seed(123)

    # 1. 层输入 (Layer inputs)
    # 形状为(2,5)，表示2个样本，每个样本5个维度（特征）
    batch_example = torch.randn(2, 5)
    print("层输入 (2个样本，每个5维):")
    print(batch_example)
    print("输入形状:", batch_example.shape)
    print("------------------------")

    # 2. 定义神经网络层
    # 包含：
    # - 全连接层(nn.Linear(5,6))：将5维输入映射到6维输出
    # - 激活函数(nn.ReLU())：对输出进行非线性变换
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    print("神经网络层结构:")
    print(layer)
    print("------------------------")

    # 3. 前向传播计算层输出 (Layer outputs)
    # 经过全连接层和ReLU激活函数后的结果
    out = layer(batch_example)

    # 4. 层输出结果
    print("层输出 (经过全连接层+ReLU后):")
    print(out)
    """
        在计算均值或方差等操作时使用 keepdim=True 参数，可以确保输出张量的维度与输入张量相同，即使该操作通过dim参数减少了张量的维度。
        例如，如果不使用 keepdim=True，返回的均值张量将是一个二维向量 [0.1324, 0.2170]，
        而使用 keepdim=True 后，返回的张量则会是一个 2×1 的矩阵 [​[0.1324], [0.2170]​]
    """
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)
    print("输出形状:", out.shape)  # 保持(2,6)，2个样本，每个6维
    print("Mean:\n", mean) # 第一个值代表第一个样本的均值，第二个值代表第二个样本的均值
    print("Variance:\n", var) # 第一个值代表第一个样本的方差，第二个值代表第二个样本的方差

    print("------------------------")
    # 5. 层归一化
    out_norm = (out - mean) / torch.sqrt(var)
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print("Normalized layer outputs:\n", out_norm)
    print("Mean:\n", mean)
    print("Variance:\n", var)


    print("------------------------")
    print("-----使用层归一化类-----")
    print("------------------------")
    ln = p4c2.LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)

def test_ch0301():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-Key-Value bias
    }
    ffn = p4c3.FeedForward(GPT_CONFIG_124M)
    #A 创建一个 batch 大小为 2 的示例输入
    x = torch.rand(2, 3, 768)
    out = ffn(x)
    print(out.shape)

def test_ch0401():
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1., 0., -1.]])
    torch.manual_seed(123) # specify random seed for the initial weights for reproducibility
    print('----------不使用快捷链连接------------')
    model_without_shortcut = p4c4.ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=False
    )
    p4c4.print_gradients(model_without_shortcut, sample_input)

    print('----------使用快捷链连接------------')
    model_with_shortcut = p4c4.ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=True
    )
    p4c4.print_gradients(model_with_shortcut, sample_input)


def test_ch0501():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024, # Context length
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-Key-Value bias
    }
    torch.manual_seed(123)
    #A 建一个形状为 [batch_size, num_tokens, emb_dim] 的输入张量
    x = torch.rand(2, 4, 768)
    block = p4c5.TransformerBlock(GPT_CONFIG_124M)
    output = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)

test_ch0501()