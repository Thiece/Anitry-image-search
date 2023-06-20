from transformers import BertModel, BertTokenizer
import torch
from scipy.spatial.distance import cosine
import numpy as np

# 初始化模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 提取标签列表的向量表示
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    # CLS token
    cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
    # Average Pooling
    avg_embedding = torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()
    # Max Pooling
    max_embedding = torch.max(outputs.last_hidden_state, dim=1).values.detach().numpy()
    
    return cls_embedding, avg_embedding, max_embedding

# 建立标签库
labels = ["label1", "label2", "label3"]  # 你的标签列表
label_embeddings = [get_embeddings(label) for label in labels]

# 自然语言查询
query = "your natural language description"

# 提取查询的向量表示
query_embeddings = get_embeddings(query)

# 搜索最相似的标签
similarities = []
for label_embedding in label_embeddings:
    similarities.append([1 - cosine(query_embedding, single_label_embedding) 
                         for query_embedding, single_label_embedding in zip(query_embeddings, label_embedding)])

# 求每种池化方式下的最大相似性
max_similarities = [np.max(similarity) for similarity in zip(*similarities)]

# 找到最大相似性对应的标签
most_similar_labels = [labels[similarities[i].index(max_similarities[i])] for i in range(3)]

print("Most similar labels: ", most_similar_labels)
