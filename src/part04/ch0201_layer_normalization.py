import torch
import torch.nn as nn

# Listing 4.2 A layer normalization class
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    """
        它作用于输入张量 x 的最后一个维度，该维度表示嵌入维度（emb_dim）​
        变量 eps 是一个小常数（epsilon）​，在归一化过程中加到方差上，以防止出现除零错误
        scale 和 shift 是两个可训练参数（与输入具有相同的维度）​
        
        大语言模型（LLM）在训练中会自动调整这些参数，以改善模型在训练任务上的性能
        这使得模型能够学习适合数据处理的最佳缩放和偏移方式
    """
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift