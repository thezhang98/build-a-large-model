# Listing 4.6 The transformer block component of GPT
from src.part03.ch0602_multi_head_attention import MultiHeadAttention
import src.part04.ch0301_feed_forward as p4c3
import src.part04.ch0201_layer_normalization as p4c2
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = p4c3.FeedForward(cfg)
        self.norm1 = p4c2.LayerNorm(cfg["emb_dim"])
        self.norm2 = p4c2.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        #A 注意力模块中的快捷连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back
        #B 前馈网络模块中的快捷链接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        #C 将原始输入加回到输出中
        x = x + shortcut
        return x

