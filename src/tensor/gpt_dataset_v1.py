# 2.6 基于torch的dataset 加載器
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class GPTDataSetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

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
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))


        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]