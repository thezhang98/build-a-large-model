import torch

# Listing 4.8 A function for the GPT model to generate text
#A idx 是当前上下文中索引的数组，形状为 (batch, n_tokens)
def generate_text_simple(model, idx, max_new_tokens, context_size): #A
    for _ in range(max_new_tokens):
        #B 若上下文长度超出支持范围，则进行裁剪。例如，若模型仅支持 5 个 token，而上下文长度为 10，仅使用最后 5 个 token 作为上下文
        idx_cond = idx[:, -context_size:]                           #B
        with torch.no_grad():
           logits = model(idx_cond)

        #C 仅关注最后一个时间步，将形状从 (batch, n_token, vocab_size) 转换为 (batch, vocab_size)
        logits = logits[:, -1, :]                                   #C
        #D probas 的形状为 (batch, vocab_size)
        # 对 logits 进行 softmax 运算，得到概率分布
        probas = torch.softmax(logits, dim=-1)                      #D
        #E idx_next 的形状为 (batch, 1)
        # 找到概率分布中概率最大的索引
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)       #E
        #F 将采样的索引追加到当前序列中，此时 idx 的形状为 (batch, n_tokens+1)
        idx = torch.cat((idx, idx_next), dim=1)                     #F

    return idx



