import torch
import src.part03.ch0402_simple_self_attention_v1 as ch0402

def test_sa_v1():
    x = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ])

    sa = ch0402.SelfAttention_v1(3, 3)
    sa.forward(x)
    print(sa.W_key)

def test_dot():
    inputs = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ])
    query = torch.tensor([0.1, 0.2, 0.3])
    res = 0.
    for idx, element in enumerate(inputs[0]):
        res += inputs[0][idx] * query[idx]
    print(res)
    print(torch.dot(inputs[0], query))
    print(inputs[0] @ query)

    """
        证明点积的运算方式是两个一维向量的对应元素相乘，然后再相加的和，也就是一个标量
        对于矩阵乘法@而言，当两个矩阵都是一维时，矩阵乘法等效于点积
    """

def test_matrix_mul():
    a = torch.tensor([
        [0.1, 0.2],
        [0.3, 0.4]
    ])
    b = torch.tensor([
        [0.5, 0.6],
        [0.7, 0.8]
    ])
    c = a @ b
    print(c)
    """
        Res = [
            [0.1900, 0.2200],
            [0.4300, 0.5000]
        ]
        0.1900 = 0.1 * 0.5 + 0.2 * 0.7
        0.2200 = 0.1 * 0.6 + 0.2 * 0.8
        ......
    """
    a = torch.tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])
    b = torch.tensor([
        [1, 2],
        [3, 4],
        [5, 6]
    ])
    c = a @ b
    print(c)
    c = b @ a
    print(c)


def test_transpose():
    a = torch.tensor([
        [0.1, 0.2],
        [0.3, 0.4]
    ])
    b = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ])
    print(a.T)
    print(b.T)
    """
    # a.T
    tensor([
        [0.1000, 0.3000], 
        [0.2000, 0.4000]
    ])
    # b.T
    tensor([
        [0.1000, 0.4000], 
        [0.2000, 0.5000], 
        [0.3000, 0.6000]
    ])
    """


test_matrix_mul()


    