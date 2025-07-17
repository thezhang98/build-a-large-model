# Listing 3.5 An efficient multi-head attention class
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out,
                 context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads                        #A
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)                   #B
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
             torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )


    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)                                      #C
        queries = self.W_query(x)                                 #C
        values = self.W_value(x)                                  #C

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)       #D
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)   #D
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim) #D

        keys = keys.transpose(1, 2)                               #E
        queries = queries.transpose(1, 2)                         #E
        values = values.transpose(1, 2)                           #E

        attn_scores = queries @ keys.transpose(2, 3)              #F
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]    #G

        attn_scores.masked_fill_(mask_bool, -torch.inf)           #H

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)     #I

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  #J
        context_vec = self.out_proj(context_vec)                  #K
        return context_vec

        #A 将投影维度缩小，以匹配期望的输出维度
        #B 使用线性层组合头部输出
        #C 张量形状：(b, num_tokens, d_out)
        #D 我们通过添加 num_heads 维度来隐式地拆分矩阵。然后展开最后一个维度，使其形状从 (b, num_tokens, d_out) 转换为 (b, num_tokens, num_heads, head_dim)
        #E 将张量的形状从 (b, num_tokens, num_heads, head_dim) 转置为 (b, num_heads, num_tokens, head_dim)
        #F 对每个注意力头进行点积运算
        #G 掩码被截断到 token 的数量
        #H 使用掩码填充注意力分数
        #I 张量形状：（b, num_tokens, n_heads, head_dim）
        #J 将多个注意力头的输出结果合并，其中输出维度 self.d_out 等于注意力头数 self.num_heads 与每个头的维度 self.head_dim 的乘积
        #K 添加一个可选的线性投影层

def test():
    #A 该张量的形状为 (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4)
    a = torch.tensor([
        [
            [
                [0.2745, 0.6584, 0.2775, 0.8573],             #A
                [0.8993, 0.0390, 0.9268, 0.7388],
                [0.7179, 0.7058, 0.9156, 0.4340]
            ],
            [
                [0.0772, 0.3565, 0.1479, 0.5331],
                [0.4066, 0.2318, 0.4545, 0.9737],
                [0.4606, 0.5159, 0.4220, 0.5786]
            ]
        ]
    ])
    # a.transpose(2, 3) = (1, 2, 4, 3) 还是一样的原因，矩阵乘法的要求
    r = a @ a.transpose(2, 3)
    print(r)
    print(r.shape)
