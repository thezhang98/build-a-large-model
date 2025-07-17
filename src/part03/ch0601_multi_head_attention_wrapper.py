# Listing 3.4 A wrapper class to implement multi-head attention
import torch
from torch import nn
import src.part03.ch0503_causale_attention as ch0503

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [ch0503.CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

def test():
    inputs = torch.tensor([
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    ])
    batch = inputs
    torch.manual_seed(123)
    context_length = batch.shape[1] # This is the number of tokens
    d_in, d_out = 3, 2
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = mha(batch)
    print(context_vecs)
    print("context_vecs.shape:", context_vecs.shape)

def test1():
    # 创建两个2x3的张量
    x = torch.randn(2, 3)  # 形状：[2, 3]
    y = torch.randn(2, 3)  # 形状：[2, 3]

    # 在第0维拼接（行方向，增加行数）
    cat0 = torch.cat([x, y], dim=0)
    print(cat0.shape)  # 输出：torch.Size([4, 3])

    # 在第1维拼接（列方向，增加列数）
    cat1 = torch.cat([x, y], dim=1)
    print(cat1.shape)  # 输出：torch.Size([2, 6])
    print(cat1)

     # 在最后一列进行拼接，增加列数
    cat2 = torch.cat([x, y], dim=-1)
    print(cat2.shape)  # 输出：torch.Size([2, 6])
    print(cat2)

