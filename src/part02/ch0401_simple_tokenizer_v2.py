# Listing 2.4 Implementing a simple text tokenizer
import re
class SimpleTokenizerV2:
    def __init__(self, vocab):
        #A 将词汇表作为类属性存储，以方便在 encode 和 decode 方法中访问
        self.str_to_int = vocab
        #B 创建一个反向词汇表，将token ID 映射回原始的文本token
        self.int_to_str = {i:s for s,i in vocab.items()}

    #C 将输入文本转换为token ID
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # 不存在的token 替换为 <|unk|>
        preprocessed = [item if item in self.str_to_int else '<|unk|>' for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    #D 将token ID 还原为文本
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        #E 在指定的标点符号前去掉空格
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text





