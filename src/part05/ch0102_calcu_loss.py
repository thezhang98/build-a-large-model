import torch
import tiktoken
from src.part04.ch0601_gpt_model import GPTModel

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    #A 我们将上下文长度从1024个token缩短到256个token
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    #B 将 dropout 设置为 0 是一种常见的做法
    "drop_rate": 0.1,
    "qkv_bias": False
}

# 新增设备管理（放在文件顶部）
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M).to(DEVICE)
inputs = torch.tensor(
    [
        [16833, 3626, 6100], # ["every effort moves",
        [40, 1107, 588]      # "I really like"]
    ], device=DEVICE
)

# Matching these inputs, the `targets` contain the token IDs we aim for the model to produce:
targets = torch.tensor(
    [
        [3626, 6100, 345 ], # [" effort moves you",
        [107, 588, 11311]   # " really like chocolate"]
    ], device=DEVICE
)

#A 禁用梯度跟踪，因为我们尚未进行训练
with torch.no_grad():
    logits = model(inputs)

# logits的形状为 (batch_size/2个批次, seq_len/3个token, vocab_size/50257=词汇表大小)，所以这个结果是模型对每个位置预测的未归一化的概率得分
print("\nLogits shape:")
print(logits.shape)

probas = torch.softmax(logits, dim=-1) # Probability of each token in vocabulary
print("\nprobas shape:")
print(probas.shape)
print(probas)

# 在词汇表维度(dim=-1)，找到概率值最大的值对应的索引
token_ids = torch.argmax(probas, dim=-1, keepdim=True) # keepdim=True 保留被操作的维度: (2,3,1) 而不是 (2,3)
print("Token IDs:\n", token_ids)

tokenizer = tiktoken.get_encoding("gpt2")
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")


## 这里开始是本节的核心：计算损失

"""
step1 拿到目标值在对应位置的概率值
    这里其实是在拿到目标值在对应位置的概率值
"""
text_idx = 0 # 第一批
target_probas_1 = probas[
    text_idx,           # 第一批文本
    [0, 1, 2],          # 前三个token的索引，一共是第三个所以这里就是全部的了
    targets[text_idx]   # 目标输出的第1批
]
print("Text 1:", target_probas_1)
text_idx = 1 # 第二批
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)

"""
step2 概率转对数概率
    - 小数相乘后只会得到一个很小的数，而对数概率可以将乘法转换为加法，从而避免数值下溢的问题。
    - 还有一个更重要的原因：
        - 概率是一个线性的波动，而对数指数级波动，这样轻微的概率波动回带来的巨大的惩罚波动
"""
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

"""
step3 计算平均值将这些对数概率合并为一个评分

"""
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

"""
step4 取反
    训练的目标就是通过更新模型权重，使平均对数概率尽可能接近 0（将在 5.2 节中实现）
    然而在深度学习中，常见做法并不是直接将平均对数概率推向 0，而是通过将负平均对数概率降低至 0 来实现
    负平均对数概率就是平均对数概率乘以 -1
"""
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)


"""
    以上是我们手动计算损失的过程
    以下是使用 PyTorch 内置函数计算损失的过程

    其结果是一致的，手动计算就是内置函数的原理
"""
logits_flat = logits.flatten(0, 1) # 把第0维和第一维合并展平，变成 (2*3, 50257)
targets_flat = targets.flatten() # 全部展平，变成 (2*3)
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)