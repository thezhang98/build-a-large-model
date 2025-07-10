# 分词器
from src.part02.ch0401_simple_tokenizer_v2 import SimpleTokenizerV2
import re

with open('../resource/txt/the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
print("单词总数: {}".format(len(raw_text)))

# 标点符号单独作为一个词
result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# 对Result的元素进行strip操作，如果strip后为空，则不保留，否则保留，最终是的result是一个不包含空格的列表
result = [item.strip() for item in result if item.strip()]

print("分词总数: {}".format(len(result)))
# print(result[:100])

# token 转 token_ids
#  1. 构建词汇表
all_words = sorted(set(result))
all_words.extend(['<|endoftext|>', '<|unk|>'])
vocab_size = len(all_words)
print("词汇表总数: {}".format(vocab_size))
#  2. 构建词汇和id映射
vocab = {token:integer for integer,token in enumerate(all_words)}

# 测试编码和解码
tokenizer = SimpleTokenizerV2(vocab)
text = """"Hello,It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
strs = tokenizer.decode(ids)
print(strs)
