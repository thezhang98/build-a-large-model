# Listing 3.3 A compact causal attention class
from this import d
import torch.nn as nn
import torch

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        #A 与之前的 SelfAttention_v1 类相比，我们添加了一个 dropout 层
        self.dropout = nn.Dropout(dropout)

        #B register_buffer 调用也是新添加的内容（buffer 会自动随模型迁移到合适的设备（CPU 或 GPU）​）
        self.register_buffer(
           'mask',
           # 生成一个对角线为 0、上三角部分全为 1 的矩阵
           # triu的第二个参数代表对角线偏移量，默认为0，意思就是以对角线为分割线,向上是1
           #    1就是对角线向上一层为分割线，向上是1
           #    -1 就是对角线向下一层为分割线，向上是1
           torch.triu(torch.ones(context_length, context_length),
           diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        """
            - 矩阵乘法
                1. 左矩阵的列数必须等于右矩阵的行数 才能进行矩阵乘法                            A列 == B行
                2. 乘积矩阵第i行第j列处的元素等于左矩阵的第i行与右矩阵的第j列对应元素乘积之和
                3. 乘积矩阵的行数等于左矩阵的行数，列数等于右矩阵的列数                          A行B列
            - 这里为什么要交换第 1 和第 2 个维度？
                - queries.shape = (batch_size, num_tokens, d_out)                           左矩阵
                - keys.shape = (batch_size, num_tokens, d_out)                              右矩阵
                - 根据矩阵乘法规则1，必须将keys维度转变为(batch_size, d_out, num_tokens)
        """
        #C 我们交换第 1 和第 2 个维度，同时保持批次维度在第1个位置（索引0）
        attn_scores = queries @ keys.transpose(1, 2)
        #D 在 PyTorch 中，带有下划线后缀的操作会在原有内存空间执行，直接修改变量本身，从而避免不必要的内存拷贝
        attn_scores.masked_fill_(
            # self.mask.bool() 将掩码转换为布尔类型（True/False）
            # [:num_tokens, :num_tokens] 裁剪矩阵
            # masked_fill_ 函数将掩码中的True值替换为指定的值：-torch.inf
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec




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
shape = inputs.shape
print(shape)
batch = inputs
d_in = 3
d_out = 3

torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
print("context_vecs:", context_vecs)