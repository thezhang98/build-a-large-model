# 2.6 基于torch的dataset 加載器
import torch
from torch.utils.data import Dataset
import tiktoken

class GPTDataSetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride, pt=False):
        self.input_ids = []
        self.target_ids = []

        #这里其实已经分词好了，固定了每个tokenid对应原始文本  
        tokenids = tokenizer.encode(txt)
        # 滑动窗口
        for i in range(0, len(tokenids) - max_length, stride):
            # 每次都会拿到基于当前位置的max_length长度的tokenids
            # 就是输入
            input_chunk = tokenids[i:i+max_length]
            # 这个则是input_chunk的下一个tokenids
            # 就是预测的输出
            target_chunk = tokenids[i+1:i+1+max_length]
            # 将tokenid转换成张量（一维） TODO：学习下张量（）
            # 那这里其实就把一个一维数组，截取max_length个，作为一个向量塞进向量数组里
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

            if pt:
                print(f"input_chunk: {input_chunk}")
                for tokenid in input_chunk:
                    print(f"decode: ", tokenizer.decode([tokenid]))
                print("==========")
            
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def test():
    tokenizer = tiktoken.get_encoding("gpt2")
    txt = "I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no great surprise to me to hear that, in the height of his glory, he had dropped his painting, married a rich widow, and established himself in a villa on the Riviera. (Though I rather thought it would have been Rome or Florence.)"
    # 创建GPTDataSetV1对象
    dataset = GPTDataSetV1(txt, tokenizer, 2, 2, true)

# test()