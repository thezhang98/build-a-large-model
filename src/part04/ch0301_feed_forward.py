import torch.nn as nn
from torch.nn import GELU
import matplotlib.pyplot as plt


# 前馈神经网络
# Listing 4.4 A feed forward neural network module
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # 这样对于输入而言，就会经过
        #   1. 输入维度为cfg["emb_dim"]，输出维度为4*cfg["emb_dim"]
        #   2. 经过GELU激活函数
        #   3. 输入维度为4*cfg["emb_dim"]，输出维度为cfg["emb_dim"]
        """
            之所以先扩展维度在缩小维度，就是在扩展维度的时候展示更多的信息
            然后呢非线性激活会分析出这些信息
            最后再缩小维度，这个时候其实那些更多的信息被压缩在了缩小维度之后的数据里了
            这样其实能够在训练过程中探索更丰富的内容
        """
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

def forward(self, x):
    return self.layers(x)