# 2.6 使用滑动窗口进行数据采样
import tiktoken
import torch
print(torch.__version__)
# 这个可在官网(https://pytorch.org/get-started/locally/)复制指定命令安装，即可使得有消息为True
print(torch.cuda.is_available())

tokenizer = tiktoken.get_encoding("gpt2")
with open('../resource/txt/the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

# TODO 还没明白为什么这么写
# 移除前50个token 
enc_sample = enc_text[50:]

context_size = 4
# 滑动窗口: 为了输出训练数据：输出-输出
for i in range(1, context_size+1):
    # 每次拿到的是前i个token，随着循环context越大；代表前文
    context = enc_sample[:i]
    # 预期，代表根据前文应该生成预期的1个后文
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

