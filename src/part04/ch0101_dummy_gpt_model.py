# Listing 4.1 A placeholder GPT model architecture class
import torch
import torch.nn as nn

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        #A 为 TransformerBlock 设置占位符
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        #B 为 LayerNorm 设置占位符
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    """
        定义了数据在模型中的流动方式：
            计算输入索引的 token 嵌入和位置嵌入，应用dropout，通过 transformer block 处理数据，应用归一化，最后通过线性输出层生成 logits
    """
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
    """
        这里有个问题哈
            你看外部调用foward方法时，都是直接`model(batch)`
            其实这里是pytorch用__call__做了底层优化，所以使你通过上述写法调用了foward方法
                这么做的原因是这个张量的计算在不同设备上要做不同的处理，所以它在__call__里面可以做前置和后置方法来做一些适配
            当你定义一个和forward参数一模一样的的方法a时，它是不会调用的。
            其实就是__call__写死了使用forward方法
        def a(self, in_idx):
            return in_idx
    """

#C 一个简单的占位类，后续将被真正的 TransformerBlock 替换
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    #D 该模块无实际操作，仅原样返回输入
    def forward(self, x):
        return x

#E 一个简单的占位类，后续将被真正的 DummyLayerNorm 替换
class DummyLayerNorm(nn.Module):
    #F 此处的参数仅用于模拟LayerNorm接口
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x

