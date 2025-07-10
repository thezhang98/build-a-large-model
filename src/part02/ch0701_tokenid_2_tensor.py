# 2.7 token 转 向量
import torch

def test():
    # 4个token
    input_ids = torch.tensor([2 ,3, 5, 1, 2])
    # 6个词汇表
    vocab_size = 6
    # 大小为3的嵌入向量：3维
    output_dim = 3
    # 随机种子 123
    torch.manual_seed(123)

    # 创建一个嵌入层 
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    # 打印嵌入层的权重
    print("初始权重：")
    print(embedding_layer.weight)

    print("token ID 3 对应的嵌入向量:")
    print(embedding_layer(torch.tensor([3])))

    print("打印input_ids对应的嵌入向量")
    print(embedding_layer(input_ids))

"""
初始权重：
tensor(
    [
        [ 0.3374, -0.1778, -0.1690],
        [ 0.9178,  1.5810,  1.3010],
        [ 1.2753, -0.2010, -0.1606],
        [-0.4015,  0.9666, -1.1481],
        [-1.1589,  0.3255, -0.6315],
        [-2.8400, -0.7849, -1.4096]
    ], 
    requires_grad=True
)
    // 6行3列：
        每一行代表词汇表中的一个token
        每一列代表向量的一个维度


token ID 3 对应的嵌入向量:
tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)
    // 可以看到嵌入向量就是权重矩阵中的一行


===============    总结    =================
== 可以看到，当tokenID相同，其输出的嵌入向量一定相同
== 确定性的tokenID嵌入可以保证在任意位置出现只要相同的token，那么其含义一定是一致的，不会出现混淆的情况。
== 但是我们假设一种这样的情况：
== - 狗咬我 vs 我咬狗
== - 小明喜欢小红 vs 小红喜欢小明
== 可以很明显的看出，咬和狗、我的关联度是一致的，那么就很有可能将狗咬我混淆成我咬狗，这样含义就直接变掉了。（语言的含义高度依赖词序）
== 假如我们能够假如位置信息(2.8位置编码)，就可以避免这种情况
===============    ===    =================
"""
test()
