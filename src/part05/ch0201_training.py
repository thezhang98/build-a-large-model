import tiktoken
import torch
from src.part02.ch0604_create_data_loader import create_data_loader_v1 as create_dataloader_v1
from src.part04.ch0601_gpt_model import GPTModel
from src.part04.ch0701_generate_text  import generate_text_simple


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())


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


def calc_loss_batch(input_batch, target_batch, model, device):
    #A 将数据传输到指定设备（如 GPU），使数据能够在 GPU 上处理。
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # model会把input的(2,256)转换成(2,256, vocab_size)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def evaluate_model(model, train_loader, val_loader, device, eval_iter):

    #A 评估阶段禁用 dropout，以确保结果稳定、可复现
    model.eval()                #A
    #B 禁用梯度跟踪，减少计算开销
    with torch.no_grad():       #B
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """
        这里就是实测下模型的文本生成能力，能更直观的监控模型训练效果
    """
    model.eval() # 实测文本生成也关掉dropout
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad(): # 禁用损失梯度计算
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " ")) # Compact print format
    model.train()


# Listing 5.3 The main function for pretraining LLMs
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    #A 初始化用于记录损失和已处理 token 数量的列表
    train_losses, val_losses, track_tokens_seen = [], [], []                        #A
    tokens_seen, global_step = 0, -1

    #B 开始主训练循环
    for epoch in range(num_epochs):                                                 #B
        model.train()
        for input_batch, target_batch in train_loader:
            #C 重置上一批次的损失梯度
            optimizer.zero_grad()                                                   #C
            #D 计算损失梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)        #D
            loss.backward()                                                         #D
            #E 使用损失梯度更新模型权重
            optimizer.step()                                                        #E
            tokens_seen += input_batch.numel()
            global_step += 1
            #F 可选的评估步骤
            if global_step % eval_freq == 0:                                        #F
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        #G 每个 epoch 结束后打印示例文本
        generate_and_print_sample(                                                  #G
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen


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
train_ratio = 0.90
with open('../resource/txt/the-verdict.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
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

#A .parameters() 方法返回模型的所有可训练权重参数
# AdamW是训练过程中的优化器，通过改进权重衰减方式，帮助减少模型复杂度，并通过惩罚较大的权重来防止过拟合
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)      #A
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=1,
    start_context="Every effort moves you", tokenizer=tokenizer
)

"""
    它这个训练是这么回事：
        从训练集中拿输入批和输出批数据，然后呢开启训练模式，就是已开启dropout等利于训练的操作
        接着把输入批数据传入模型跑出一个结果集(forward)，把这个结果集和输出批数据进行损失计算
        把这个损失进行反向传播(backward,就是计算每个参数对损失的贡献度--梯度(张量 和参数参数形状一致), 梯度的大小决定了参数更新幅度, 梯度的方向决定了参数更新的方向)，再更新模型参数(优化器保存了模型所有的可训练参数)

        更新参数后 其实一个批次就结束了 但是一轮训练会有多个批次 每个批次开始时都需要把上次更新的梯度清零 否则会累加起来
"""


