# 2.8 位置编码
import torch
from src.part02.ch0604_create_data_loader import create_data_loader_v1
import tiktoken


def printDataload(dataloader):
    # 计算dataloader的总批次数
    total_batches = len(dataloader)
    print(f"DataLoader总批次数: {total_batches}")

    # 计算总样本数
    total_samples = len(dataloader.dataset)
    print(f"总样本数: {total_samples}")

    # 验证每个批次的形状
    for i, (inputs_batch, targets_batch) in enumerate(dataloader):
        print(f"批次 {i+1}/{total_batches}: inputs.shape = {inputs_batch.shape}, targets.shape = {targets_batch.shape}")
        if i == 0:  # 只打印第一个批次的内容
            print(f"第一个批次的Token IDs:\n{inputs_batch}")
        if i >= 2:  # 只打印前3个批次
            break

def printTokenEmbedding(token_embeddings, inputs):
    # 打印 token_embeddings 的具体内容
    print("\n===== token_embeddings 内容详情 =====")

    # 打印整体形状和数据类型
    print(f"嵌入向量张量形状: {token_embeddings.shape}")  # [8, 4, 256]
    print(f"数据类型: {token_embeddings.dtype}")  # 通常为 torch.float32

    tokenizer = tiktoken.get_encoding("gpt2")
    # 打印第一个样本的所有 token 嵌入
    print("\n第一个样本的4个token的嵌入向量（前10维）:")
    for i in range(token_embeddings.shape[1]):  # 第一个样本 遍历4个token
        token_id = inputs[0, i].item()  # 获取对应的token ID
        token_text = tokenizer.decode([token_id])  # 解码为文本
        embedding = token_embeddings[0, i, :10].detach().numpy()  # 取前10维
        print(f"  Token {i+1}: ID={token_id}, 文本='{token_text}', 嵌入向量前10维: {embedding}")

    # 可视化嵌入向量的分布（简化为统计值）
    print("\n嵌入向量的统计特征（以第一个样本的第一个token为例）:")
    first_embedding = token_embeddings[0, 0].detach().numpy()
    print(f"  最小值: {first_embedding.min():.4f}")
    print(f"  最大值: {first_embedding.max():.4f}")
    print(f"  均值: {first_embedding.mean():.4f}")
    print(f"  标准差: {first_embedding.std():.4f}")

    # 对比不同token嵌入的相似性
    print("\n不同token嵌入的余弦相似度:")
    from torch.nn.functional import cosine_similarity

    # 计算第一个样本中前两个token的嵌入相似度
    sim = cosine_similarity(
        token_embeddings[0, 0].unsqueeze(0), 
        token_embeddings[0, 1].unsqueeze(0)
    ).item()
    print(f"  第1个token与第2个token的相似度: {sim:.4f}")



with open('../resource/txt/the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
max_length = 4

vocab_size = 50257  # 词汇表50257
output_dim = 256    # 嵌入维度
# 1.创建一个嵌入层 50257 * 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)


# 2. 使用滑动窗口进行数据采样
dataloader = create_data_loader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
# printDataload(dataloader)
"""
dataloader的大小是这样计算的：
    - 所有文本首先会被分词器分成tokenid和token的映射，并且会被固定。
    - 接着被滑动窗口进行数据采样：将分词后的tokenid按照窗口(max_length)大小分为一组并转为张量(形状为[max_length],输入和输出都是一样的逻辑)；最后被转为dataset
        所以dataset迭代器后每次返回两个张量，一个窗口大小的输入tokenid，一个窗口大小的输出tokenid
    - dataset被作为参数传入了DataLoader，因为每批次是8个token(样本)，对于dataloader而言，每个输入/出批次都是一个8(行，每批次的数量=batch_size) * 4(列，滑动窗口大小的数量=max_length)
        批次的数量 = 总token数 / 每批次的数量
        dataloader = [
            # 第一批次 
            [
                [[x,x,x,x], [x,x,x,x], ....], # batch_size个输入tokenid张量，每个张量是4个tokenid   形状: [8, 4]
                [[y,y,y,y], [y,y,y,y], ....], # batch_size个输出tokenid张量，每个张量是4个tokenid   形状: [8, 4]
            ],
            # 第二批次
            ......
        ]
"""
data_iter = iter(dataloader)
# dataload会返回两个数组，一个是输入，一个是输出
inputs, target = next(data_iter)
# print("Token IDs:\n", inputs)
# 就是数据的矩阵形状：8个样本 * 4个token
# print("\nInputs shape:\n", inputs.shape)

# 3. 使用嵌入层将tokenid转换为嵌入向量
token_embeddings = token_embedding_layer(inputs)
# 形状: [8, 4, 256]
# print("token_embeddings shape:\n", token_embeddings.shape)
# printTokenEmbedding(token_embeddings, inputs)

# 4. 创建位置嵌入层
context_length = max_length
# 形状: [4, 256]
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
# print(pos_embeddings.shape)

# 4.1 将位置向量添加到每个4*256维的token嵌入张量中
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)

"""
# 形状: [2, 3, 4]
token_embeddings = [
    # 样本1
    [
        [0.1, 0.2, 0.3, 0.4],  # 第1个token的嵌入（如 "I"）
        [0.5, 0.6, 0.7, 0.8],  # 第2个token的嵌入（如 "love"）
        [0.9, 1.0, 1.1, 1.2]   # 第3个token的嵌入（如 "you"）
    ],
    # 样本2
    [
        [1.3, 1.4, 1.5, 1.6],  # 第1个token的嵌入（如 "He"）
        [1.7, 1.8, 1.9, 2.0],  # 第2个token的嵌入（如 "likes"）
        [2.1, 2.2, 2.3, 2.4]   # 第3个token的嵌入（如 "cats"）
    ]
]

# 形状: [3, 4]
pos_embeddings = [
    [0.0, 0.0, 0.0, 0.0],  # 第1个位置的嵌入
    [0.1, 0.1, 0.1, 0.1],  # 第2个位置的嵌入
    [0.2, 0.2, 0.2, 0.2]   # 第3个位置的嵌入
]

# 计算过程
# input_embeddings = token_embeddings + pos_embeddings
input_embeddings = [
    # 样本1
    [
        [0.1+0.0, 0.2+0.0, 0.3+0.0, 0.4+0.0],  # 样本1的第1个token（位置1）
        [0.5+0.1, 0.6+0.1, 0.7+0.1, 0.8+0.1],  # 样本1的第2个token（位置2）
        [0.9+0.2, 1.0+0.2, 1.1+0.2, 1.2+0.2]   # 样本1的第3个token（位置3）
    ],
    # 样本2
    [
        [1.3+0.0, 1.4+0.0, 1.5+0.0, 1.6+0.0],  # 样本2的第1个token（位置1）
        [1.7+0.1, 1.8+0.1, 1.9+0.1, 2.0+0.1],  # 样本2的第2个token（位置2）
        [2.1+0.2, 2.2+0.2, 2.3+0.2, 2.4+0.2]   # 样本2的第3个token（位置3）
    ]
]

# 最终结果
input_embeddings = [
    # 样本1
    [
        [0.1, 0.2, 0.3, 0.4],  # 位置1的 "I"
        [0.6, 0.7, 0.8, 0.9],  # 位置2的 "love"
        [1.1, 1.2, 1.3, 1.4]   # 位置3的 "you"
    ],
    # 样本2
    [
        [1.3, 1.4, 1.5, 1.6],  # 位置1的 "He"
        [1.8, 1.9, 2.0, 2.1],  # 位置2的 "likes"
        [2.3, 2.4, 2.5, 2.6]   # 位置3的 "cats"
    ]
]
"""
