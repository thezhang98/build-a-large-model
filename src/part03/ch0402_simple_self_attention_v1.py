# Listing 3.4.2 A compact self-attention class
import torch.nn as nn
import torch

# 通过继承nn.Module来自定义自注意力类
class SelfAttention_v1(nn.Module):
    # 输入维度d_in
    # 输出维度d_out
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        """
        - nn.Parameter
            > nn.Parameter是torch.Tensor的子类，被自动注册到模型的参数列表中
            > 在训练过程中，优化器会直接更新这些参数
            将参数中的矩阵标记为可训练参数，纳入模型优化范围
        - torch.rand(d_in, d_out)
            > 穿件一个随机矩阵，形状为(d_in, d_out)
        """

        """
        - 三个可训练的权重参数矩阵
            > 作用：它们使模型（特别是模型内部的注意力模块）能够学习生成“优质”的上下文向量
            > 意义：定义网络层之间的连接关系
        """
        # 
        # 
        #
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))


    # 前向传播方法 , x是输入
    def forward(self, x):
        """
        - @
            > 矩阵乘法，对应元素相乘后求和
        - T 
            > T 是转置操作，将矩阵的行和列互换,参见 test.py => test_transpose()
        - attn_scores
            > 计算注意力分数：查询与k的转置相乘得到的相似度分数
        - attn_weights
            > 计算注意力权重：对注意力分数进行缩放(除以k向量维度的平方根)，然后应用softmax函数归一化
            > 意义：用于确定上下文向量对输入文本的不同部分的依赖程度，即神经网络对输入不同部分的关注程度
        - context_vec
            > 计算上下文向量：注意力权重与值的矩阵乘积

        - 可以和ch0302_test_sample_self_attention.py 对比注意力得分、注意力权重和上下文向量有什么不同

        - TODO: 
            > 这里要理解一下为什么q,k,v,attn_scores和attn_weights,context_vec 为什么要这么算
            > 为什么需要矩阵乘法，或者说为什么是矩阵乘法
            > 归一化为什么这么算

        """
        # q,k,v 是输入与q,k,v权重矩阵的乘积
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec