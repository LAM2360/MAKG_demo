import re
from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer(model_name_or_path='model_path')

with open("file_path") as f:
    text = f.read()

# Splitting the text on '。'  '；' '?' and '\n'
single_sentences_list = re.split(r'[。；？！\n]+', text)
print (f"{len(single_sentences_list)} senteneces were found")

sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(single_sentences_list)]

# print(sentences[:3])

def combine_sentences(sentences, buffer_size=1):
    combined_sentences = [
        ' '.join(sentences[j]['sentence'] for j in range(max(i - buffer_size, 0), min(i + buffer_size + 1, len(sentences))))
        for i in range(len(sentences))
    ]   
    # 更新原始字典列表，添加组合后的句子
    for i, combined_sentence in enumerate(combined_sentences):
        sentences[i]['combined_sentence'] = combined_sentence

    return sentences

sentences = combine_sentences(sentences, buffer_size=1)

embeddings = model.encode([x['combined_sentence'] for x in sentences])

for i, sentence in enumerate(sentences):
    sentence['combined_sentence_embedding'] = embeddings[i]


def cosine_similarity(vec1, vec2):
    """Calculate the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        similarity = cosine_similarity(embedding_current, embedding_next)
        distance = 1 - similarity
        distances.append(distance)
        sentences[i]['distance_to_next'] = distance
    return distances, sentences

distances, sentences = calculate_cosine_distances(sentences)

# 初始化起始索引
start_index = 0

chunks = []

breakpoint_percentile_threshold = 90
breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold) # 如果你想更多的分块，降低百分位阈值

indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold] 

for index in indices_above_thresh:
    end_index = index

    group = sentences[start_index:end_index + 1]
    combined_text = ' '.join([d['sentence'] for d in group])
    chunks.append(combined_text)
    
    start_index = index + 1

if start_index < len(sentences):
    combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
    chunks.append(combined_text)

for i, chunk in enumerate(chunks):
    print (f"Chunk #{i}")
    print (chunk)
    print ("\n")