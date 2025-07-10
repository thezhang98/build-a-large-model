# 使用bep来作为分词算法，bep本质是基于统计概率的分词算法
from importlib.metadata import version
import tiktoken

print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

# with open('../resource/txt/the-verdict.txt', 'r', encoding='utf-8') as f:
#     raw_text = f.read()
# enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))


s = "hello world,and i say you are champion，你好"
enc_text = tokenizer.encode(s)
enc_text.extend([1024,9999,3,4,5])
print(enc_text)
print(tokenizer.decode(enc_text))

