import torch

def calculate_sample_attention_scores(query, inputs, log = False):
    # 创建一个空的初始化张量，形状由inputs.shape[0]决定，因为inputs.shape[0]是(6,3)，所以这个张量是(6)
    attn_scores_2 = torch.empty(inputs.shape[0])
    
    # 这里其实就是x^2和所有token计算点积，得到一个6维的向量，每一个元素都是原输入token与x^2的点积
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)
    if log:
        print(inputs.shape)
        print(attn_scores_2.shape)
        print(attn_scores_2)
        print("=====================")
    return attn_scores_2

def calculate_context_vector(query, inputs, attn_weights_2, log = False):
    # 创建一个全零张量作为context vector，其维度维同query，query是3维
    context_vec_2 = torch.zeros(query.shape)
    for i,x_i in enumerate(inputs):
        # 计算x^2对x_i的上下文向量：通过 x^2对于x_i的权重 乘以 x_i元素向量
        tmp = attn_weights_2[i]*x_i
        context_vec_2 += tmp
        if log:
            print(f"{i}: {x_i} * {attn_weights_2[i]} => {tmp}")
    # if log:
    #     print(context_vec_2)
        # print("=====================")
    return context_vec_2

def test_softmax(attn_scores_2, log = False):
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    if log:
        print("Attention weights:", attn_weights_2_tmp)
        print("Sum:", attn_weights_2_tmp.sum())
        print("=====================")
    return attn_weights_2_tmp


def test_muilt_context_vector(inputs):
    eles = [
        "Your",
        "journey",
        "starts",
        "with",
        "one",
        "step",
    ]
    inputs = torch.tensor([
        [0.43, 0.15, 0.89], # Your     (x^1)
        [0.55, 0.87, 0.66], # journey  (x^2)
        [0.57, 0.85, 0.64], # starts   (x^3)
        [0.22, 0.58, 0.33], # with     (x^4)
        [0.77, 0.25, 0.10], # one      (x^5)
        [0.05, 0.80, 0.55]  # step     (x^6)
    ])
    for i, x_i in enumerate(inputs):
        query = inputs[i] # 第i+1个输入:x^2 token 用作查询向量-q
        ele = eles[i]
        # step1 计算x^2对所有token的注意力分数
        attn_scores_2 = calculate_sample_attention_scores(query, inputs)
        print(f"- step1: {ele} 对所有token的注意力分数")
        print(attn_scores_2)

        # step2 归一化注意力分数得到注意力权重, 也就是x^2对所有token的注意力权重
        attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
        print(f"- step2: {ele} 对所有token的注意力权重")
        print(attn_weights_2)

        # step3 计算上下文向量，把x^2对所有token的注意力整合在一起了，因为一个元素是3维，所以最后输出一个3维张量，张量的每一个元素都包含了x^2对所有token的注意力
        # 这是一种信息聚焦，的确会遗失一些不重要的信息，所以是无法反推回去的
        # 上下文向量是语义摘要而非数据副本
        context_vec_2 = calculate_context_vector(query, inputs, attn_weights_2, True)
        print(f"- step3: {ele} 对所有token的上下文向量")
        print(context_vec_2)

        print("==========                                                         ===========")
    return

inputs = torch.tensor([
    [0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2)
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]  # step     (x^6)
])

test_muilt_context_vector(inputs)



