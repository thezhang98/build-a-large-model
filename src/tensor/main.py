# 2.6 使用torch的测试
import tiktoken
from torch.utils.data import DataLoader, dataloader
from gpt_dataset_v1 import GPTDataSetV1

def create_data_loader_v1(txt, batch_size=1, max_length=4, stride=1, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataSetV1(txt, tokenizer, max_length, stride)
    # drop_last=True 表示如果最后一个batch的数据量小于batch_size，那么就丢弃掉这个batch,以防止训练期间损失峰值
    # num_workers=0 表示不使用CPU多线程
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader

def test_step_and_context_size(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False):
    dataloader = create_data_loader_v1(raw_text, batch_size, max_length, stride, shuffle)
    # 把数据迭代器迭代出来
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    second_batch = next(data_iter)
    print(second_batch)
    print("==========")

with open("../resource/txt/the-verdict.txt") as f:
    raw_text = f.read()

test_step_and_context_size(raw_text)

# - max_length 测试
# max_length 决定了滑动窗口的大小，也就决定了每组输入输出的长度
# test_step_and_context_size(raw_text, batch_size=1,max_length=5)
# test_step_and_context_size(raw_text, batch_size=1,max_length=6)

# - stride 测试
# stride 决定了滑动窗口的步长，也就决定了每组输入输出的偏移
# test_step_and_context_size(raw_text, stride=2)
# test_step_and_context_size(raw_text, stride=3)


# - 测试batch_size
# TODO 还没能理解输出的到底是几维张量？
test_step_and_context_size(raw_text, batch_size=2, max_length=4, stride=4) # max_length = stride, 这样就可以保证不会漏掉token
test_step_and_context_size(raw_text, batch_size=3, max_length=4, stride=4)
