import tiktoken
import torch
from src.part02.ch0604_create_data_loader import create_data_loader_v1 as create_dataloader_v1
from src.part04.ch0601_gpt_model import GPTModel



with open('../resource/txt/the-verdict.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()

tokenizer = tiktoken.get_encoding("gpt2")


total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

torch.manual_seed(123)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
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

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

print("Train loader:")
for x, y in train_loader:
    print(x.shape, y.shape)


print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

def calc_loss_batch(input_batch, target_batch, model, device):
    #A 将数据传输到指定设备（如 GPU），使数据能够在 GPU 上处理。
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # model会把input的(2,256)转换成(2,256, vocab_size)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


# 计算训练集和验证集的损失
# Listing 5.2 Function to compute the training and validation loss
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        #A 如果没有指定批次数，将自动遍历所有批次
        num_batches = len(data_loader)                                    #A
    else:
        #B 若批次数超过数据加载器的总批次数，则减少批次数使其与数据加载器的批次数相匹配
        num_batches = min(num_batches, len(data_loader))                  #B
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            #C 每个批次的损失求和
            total_loss += loss.item()                                     #C
        else:
            break
    #D 对所有批次的损失取平均值
    return total_loss / num_batches                                       #D


#A 如果你的设备配备了支持 CUDA 的 GPU，LLM 将自动在 GPU 上进行训练，无需更改代码
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #A
model = GPTModel(GPT_CONFIG_124M).to(device)
#B 因为当前不在训练，为提高效率，关闭梯度跟踪
with torch.no_grad():                                                 #B
    #C 通过 device 设置确保数据与 LLM 模型加载到同一设备上
    train_loss = calc_loss_loader(train_loader, model, device)        #C
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)
print("Validation loss:", val_loss)

