import torch
import torch.nn as nn
import src.part04.ch0101_dummy_gpt_model as p4c1
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


test_ch0101()