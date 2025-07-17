# 因果注意力掩码
import torch
from torch.nn.modules import dropout
import src.part03.ch0402_simple_self_attention_v2 as ch0402_v2

def masked_self_attention_v1(sa_v2, inputs):
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    # keys是(2,3) 所以keys.T是(3,2)
    # queries是(2,3) 所以queries @ keys.T = (2,3) @ (3,2) = (2,2) 即 A行B列
    attn_scores = queries @ keys.T
    # print("queries", queries)
    # print("keys", keys)
    # print("attn_scores\n", attn_scores)
    
    # step-1 得到原始注意力权重矩阵
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
    print("attn_weights\n", attn_weights)

    """
        - 如何使用掩码
            > 这里掩码是通过将之前归一化后的注意力权重矩阵与掩码矩阵相乘来实现的
            > 也就是要做两遍归一化
        - 为什么要用掩码
            > 因果注意力掩码就是为了让当前元素无法访问之后的元素，这样模型在训练阶段就不会通过阅读后续的元素进行作弊了，同时也保持了推理与训练的一致性
    """
    # step-2 掩码
    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print("mask_simple\n", mask_simple)
    
    """
        - 原始注意力权重矩阵
                    Your	journey	starts	with	one	    step
            Your	0.19	0.16	0.16	0.15	0.17	0.15
            journey	0.20	0.16	0.16	0.14	0.16	0.14
            starts	0.20	0.16	0.16	0.14	0.16	0.14
            with	0.18	0.16	0.16	0.15	0.16	0.15
            one	    0.18	0.16	0.16	0.15	0.16	0.15
            step	0.19	0.16	0.16	0.15	0.16	0.15
        
        - 掩码后的注意力权重矩阵
                    Your	journey	starts	with	one	    step	
            Your	1.0	    0	    0	    0	    0	    0
            journey	0.55	0.44	0	    0	    0	    0	
            starts	0.38	0.30	0.31	0	    0	    0	
            with	0.27	0.24	0.24	0.23	0	    0	
            one	    0.21	0.19	0.19	0.18	0.19	0	
            step	0.19	0.16	0.16	0.15	0.16	0.15	
            
        可以看到，对于"Your"来讲，它只能看到自己，不能看到其他元素
    """
     # step-3 重新计算注意力权重矩阵： 理解矩阵相乘就能理解为什么要使对角线以上被掩码
    masked_simple = attn_weights * mask_simple # 这里用元素相乘，即可使原矩阵被掩码
    print("掩码后的注意力权重矩阵\n", masked_simple)
    # step-4 归一化
    row_sums = masked_simple.sum(dim=1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    print("归一化后的注意力权重矩阵\n",masked_simple_norm)


def masked_self_attention_v2(sa_v2, inputs):
    """
        与v1的区别在于：
            这里的掩码直接是被填充inf的注意力得分
    """
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    attn_scores = queries @ keys.T
    
    # step-1 得到原始注意力权重矩阵
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
    print("attn_weights\n", attn_weights)

    # step-2 掩码
    context_length = attn_scores.shape[0]
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print(masked)

    # step-3 归一化
    attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
    print("归一化后的注意力权重矩阵\n",attn_weights)
    return attn_weights



def masked_self_attention_v3(sa_v2, inputs):
    torch.manual_seed(123)
    # 我们使用的dropout率为0.5
    dropout = torch.nn.Dropout(0.5)
    attn_weights =masked_self_attention_v2(sa_v2, inputs)
    print("masked_self_attention_v3\n")
    print("dropout后的权重矩阵\n", dropout(attn_weights))

inputs = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])
sa_v2 = ch0402_v2.SelfAttention_v2(d_in=3, d_out=3)
# masked_self_attention_v1(sa_v2, inputs)
masked_self_attention_v2(sa_v2, inputs)
masked_self_attention_v3(sa_v2, inputs)
