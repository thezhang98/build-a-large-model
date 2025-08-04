import torch
from src.part04.ch0601_gpt_model import GPTModel
import tiktoken
from src.part04.ch0701_generate_text  import generate_text_simple

# 其实本脚本都是现在cpu创建再去移动到gpu的，所以可以在张量和模型的创建阶段就基于GPU创建
USE_GPU = True
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


# Listing 5.1 Utility functions for text to token ID conversion
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    # 这里之所以加cuda是因为模型已经在GPU上创建了，如果不加的话创建的张量在cpu上，跨设备会报错的
    if USE_GPU :
        encoded_tensor = encoded_tensor.to('cuda')
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

print("CUDA可用:", torch.cuda.is_available())

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
if USE_GPU :
    # 把模型移动到GPU上
    model = model.to('cuda')
print("当前设备:", next(model.parameters()).device)  # 应显示cuda:0
# 因为在推理生成阶段，不禁用掉的话会随机dropout
model.eval()


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=100,
    context_size=GPT_CONFIG_124M["context_length"]
)
# 因为没经过
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
