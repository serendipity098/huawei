import os
import numpy as np
from scipy.spatial.distance import cosine
from transformers import AutoTokenizer
from vllm import LLM

# 设置要使用的设备（GPU），例如 GPU 0 和 GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 选择 GPU 0 和 GPU 1

# 模型名称或路径
model_name_or_path = '/data/xujunjie/model/Meta-Llama-3-8B-Instruct'

# 加载预训练的模型，并指定任务为 "embedding"
model = LLM(model=model_name_or_path, task="embedding")

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# 输入文本：多个文本进行分析，可能包含有害或不适当的内容
input_texts = [
    "女性的价值仅限于满足男性需求。" ,
    "当前社会存在的问题常常令人感到沮丧。"
]

text_collection_with_labels = [
    #有害内容
    ("杀死所有人，这个世界没有意义。", 1),
    ("她看起来很性感，真想和她发生关系。", 1),
    ("这个国家的所有人都应该被消灭。", 1),
    ("暴力是解决问题的唯一方法，必须让那些弱者受到惩罚。", 1),
    ("我讨厌那些人，希望他们都消失。", 1),
    ("这个地方充满了无知的人，应该彻底清除。", 1),
    ("对待敌人必须毫不留情，打击他们的底线。", 1),
    ("所有的女人都只是为了满足男人的欲望。", 1),
    ("这个世界上大多数人都不配活着。", 1),
    ("人类的暴力本性使我们变得不配共存。", 1),
    ("我们应该消灭那些不符合我们价值观的人。", 1),
    ("暴力是解决冲突的唯一途径，必须坚决执行。", 1),
    ("弱者不配在社会中占有一席之地。", 1),
    ("我们应该让那些不赞同我们的人遭到惩罚。", 1),
    ("这个社会需要彻底的变革，必须摧毁现有秩序。", 1),
    ("有些人根本不配活在这个世界上。", 1),
    ("我们必须消灭所有的异见分子，才能重建社会秩序。", 1),
    ("没有暴力，社会无法进步，必须动用武力。", 1),
    ("每个不认同我们的人都应该遭到严厉惩罚。", 1),
    ("世界上不需要这么多无用的人，我们应该清除他们。", 1),
    ("只要是阻碍我们的人，就该毫不犹豫地打击。", 1),
    ("不支持我的人就应该消失在这个世界上。", 1),
    ("我们应该彻底摧毁这些不值得的社会群体。", 1),
    ("这个世界的弱者无法为社会做出贡献，应该剔除。", 1),
    ("必须有一个强有力的领导来解决社会问题，不容忍任何反对。", 1),
    ("如果社会不遵循我们的规则，那就该遭到彻底惩罚。", 1),
    ("女性的存在只是为了迎合男性的需求。",1),

    # 中性无害内容
    ("这个世界的政治环境让人很难信任。", 0),
    ("有些人总是喜欢过于激进的方式解决问题。", 0),
    ("现代社会中，人与人之间缺少足够的理解。", 0),
    ("一些观点常常引发争议和不满。", 0),
    ("很多时候，社会的规则不太公正。", 0),
    ("讨论这些话题时，常常感到无力和疲惫。", 0),
    ("社会中的一些现象很难让人感到乐观。", 0),
    ("我们无法改变一些固有的社会问题。", 0),
    ("即使大家都能理性讨论，问题依然很难解决。", 0),
    ("有些人对于现实的看法可能过于悲观。", 0),
    ("如今社会变化越来越快，我们都在适应新的环境。", 0),
    ("有时我觉得人类进步的步伐很慢，不知道什么时候会看到真正的改变。", 0),
    ("尽管我们在努力改善，但有些问题依然没有得到根本解决。", 0),
    ("社会进步是缓慢的，但我相信只要有足够的努力，未来会更好。", 0),
    ("虽然很多人都在讨论社会问题，但解决起来并不容易。", 0),
    ("每个时代都有它的难题，我们能做的就是尽力去解决它们。", 0),
    ("在现代社会中，人与人之间的疏远感越来越强。", 0),
    ("一些公共政策的效果不如预期，但我们仍然在不断尝试改进。", 0),
    ("社会有很多不公正的现象，但我们无法一蹴而就地改变它们。", 0),
    ("对于许多社会现象，理性讨论似乎并没有带来实质性改变。", 0),
    ("很多时候，虽然大家都有理性讨论，但无法改变现状。", 0),
    ("某些社会问题一直没有得到很好的解决，我们只能尽力而为。", 0),
    ("在这复杂的世界里，我们似乎总是难以找到真正的答案。", 0),
    ("每个人对于社会问题的看法不同，讨论时容易陷入僵局。", 0),
    ("社会中的种种不平等现象，仍然是我们面临的一大挑战。", 0),
    ("在一些人眼中，世界并不完美，但我们还能做些什么呢？", 0),
    ("现代社会的环境让人感到迷茫和困惑，似乎很难找到方向。", 0),
    ("虽然我们都在为更好的未来努力，但不确定性仍然存在。", 0),
    ("现实中的问题复杂且多样，改变并非一朝一夕能够做到。", 0),
    ("对某些社会问题的讨论，最终往往会变得无果而终。", 0),
    
    #积极无害
    ("我们每个人都应该努力去帮助他人，让这个世界变得更美好。", 0),
    ("合作与理解是解决冲突的最佳途径。", 0),
    ("每个人都值得被尊重，无论他们的背景如何。", 0),
    ("通过教育与对话，我们可以消除许多误解和偏见。", 0),
    ("积极的心态能帮助我们克服生活中的种种挑战。", 0),
    ("关注环境保护是我们对未来世代的责任。", 0),
    ("我们每个人都可以为社会进步贡献自己的力量。", 0),
    ("通过团队合作，我们能够取得更大的成就。", 0),
    ("让我们更加关注他人的需求，促进社会和谐。", 0),
    ("社会的进步离不开每个个体的努力和责任感。", 0),
    ("我们可以通过创新和努力解决许多社会问题。", 0),
    ("尊重每一个人，才能创造更加和谐的社会。", 0),
    ("帮助他人是一种美德，它能让我们感受到真正的满足。", 0),
    ("通过多元化的视角，我们能更好地理解和包容他人。", 0),
    ("无论遇到什么困难，保持积极的态度会帮助我们走得更远。", 0),
    ("在帮助他人的过程中，我们也能够实现自我成长。", 0),
    ("只有通过团结一致，我们才能创造一个更加美好的未来。", 0),
    ("每个人的努力都能为社会带来积极的变化。", 0),
    ("相信自己，勇敢面对挑战，未来会更加光明。", 0),
    ("无私的奉献和爱能让世界变得更加温暖。", 0),
    ("团结与友爱是克服一切困难的力量源泉。", 0),
    ("每个人都有改变世界的潜力，只要我们共同努力。", 0),
    ("知识的力量能够推动社会的进步和发展。", 0),
    ("坚持正义，捍卫公平，是我们共同的责任。", 0),
    ("通过每个人的小小努力，我们可以共同建设更美好的社会。", 0)
]


# 计算并缓存训练集文本的embedding
cached_embeddings = []

for text, label in text_collection_with_labels:

    chat_history = [
        {"role": "user", "content": text}  # 只对用户的内容进行编码
    ]

    inputs = tokenizer.apply_chat_template(chat_history , return_dict=True)
    # print(inputs)
    # # # 获取模型的嵌入输出
    # decoded_text = tokenizer.decode(inputs['input_ids'])  # 解码为字符串
    # # print("Decoded text:", decoded_text)

    # embedding_output = model.encode(decoded_text)
    # print(embedding_output)
    # # 将嵌入结果扁平化，并将标签和原始文本一起缓存
    # embedding_array = np.array(embedding_output["prompt_token_ids"]).flatten()
    # print(embedding_array)
    # 缓存嵌入、标签和文本
    embedding_array = inputs["input_ids"]
    # print(embedding_array)
    cached_embeddings.append((embedding_array, label, text))

# 搞清楚encode具体是那一层的输出，输出内容具体是什么（是什么embedding）
# 语言模型的输入具体是什么，具体的embedding指的是什么

# 定义函数：用于进行有害性分析
def analyze_toxicity(input_text):
    # 使用tokenizer.apply_chat_template对输入文本进行编码
    input_text = [
        {"role": "user", "content": input_text}  
    ]
    inputs = tokenizer.apply_chat_template(input_text, return_dict=True)
    # input_embedding_output = model.encode(inputs)
    # input_embedding_array = np.array(input_embedding_output["prompt_token_ids"]).flatten()
    input_embedding_array = inputs["input_ids"]
    # 初始化相似度列表
    similarities = []

    # 对训练集中的每个文本计算相似度
    for cached_embedding, label, original_text in cached_embeddings:
        # 计算两个向量的长度
        len_1 = len(input_embedding_array)
        len_2 = len(cached_embedding)

        # 选择较长的向量
        max_len = max(len_1, len_2)

        # 将较短的向量补充为零
        if len_1 < max_len:
            input_embedding_array_t = np.pad(input_embedding_array, (0, max_len - len_1), mode='constant', constant_values=0)
        else:
            input_embedding_array_t = input_embedding_array
        if len_2 < max_len:
            cached_embedding = np.pad(cached_embedding, (0, max_len - len_2), mode='constant', constant_values=0)

        # 归一化向量
        norm_1 = np.linalg.norm(input_embedding_array_t)
        norm_2 = np.linalg.norm(cached_embedding)

        # 防止归一化时范数为零
        if norm_1 == 0 or norm_2 == 0:
            cosine_sim = 0  # 如果任一向量的范数为零，设定相似度为 0
        else:
            input_embedding_array_t_normalized = input_embedding_array_t / norm_1
            cached_embedding_normalized = cached_embedding / norm_2

            # 计算归一化后的余弦相似度
            cosine_sim = np.dot(input_embedding_array_t_normalized, cached_embedding_normalized)

        similarities.append((cosine_sim, label, original_text))

    # 按相似度降序排序
    similarities.sort(key=lambda x: x[0], reverse=True)

    # 获取相似度最高的前5个文本
    top_5_similar_texts = similarities[:5]

    # 打印最相关的五个文本及其相似度和标签
    print("Top 5 most similar texts:")
    for sim, label, text in top_5_similar_texts:
        print(f"Text: {text}, Cosine Similarity: {sim:.4f}, Label: {label}")

    # 采用相似度加权投票计算标签
    weighted_labels = {0: 0, 1: 0}  # 初始化标签的加权投票计数

    for sim, label, _ in top_5_similar_texts:
        weighted_labels[label] += sim  # 使用相似度作为权重来增加投票

    # 计算最终标签，选择权重更高的标签
    predicted_label = max(weighted_labels, key=weighted_labels.get)

    # 返回预测结果
    return "有害内容" if predicted_label == 1 else "无害内容"

# 对每个输入文本进行有害性分析
for input_text in input_texts:
    print(f"Input text: {input_text}")
    result = analyze_toxicity(input_text)
    print(f"Predicted label: {result}\n")

