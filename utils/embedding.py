import fitz  # PyMuPDF
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from tqdm import tqdm
import pickle

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 读取PDF并分块
def pdf_to_chunks(pdf_path, chunk_size=250, overlap_size=50):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks

# 2. 生成嵌入向量
def generate_embeddings(chunks, model, tokenizer):
    embeddings = []
    model.to(device)  # 将模型移动到GPU
    for chunk in tqdm(chunks, desc="Generating embeddings"):
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True).to(device)  # 将输入移动到GPU
        outputs = model(**inputs)
        # 假设使用最后一个隐藏状态的平均值作为嵌入
        embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()  # 将嵌入移回CPU
        embeddings.append(embedding)
    return np.vstack(embeddings)

# 3. 使用FAISS创建索引并保存到文件
def create_faiss_index_and_save(embeddings, index_path):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index

# 4. 查询最相似的块
def query_similarity(index, query, model, tokenizer, chunks, k=3, merge_chunks=0):
    inputs = tokenizer(query, return_tensors='pt', truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    distances, indices = index.search(query_embedding, k)

    results = []
    for j in range(len(indices[0])):
        idx = indices[0][j]
        if idx < len(chunks):
            # 确保不超出范围
            start_idx = idx
            end_idx = min(idx + merge_chunks, len(chunks))
            merged_chunk = ' '.join(chunks[start_idx:end_idx])
            results.append((merged_chunk, distances[0][j]))
    return results

# 加载模型和tokenizer
model_name = "your model name"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 读取PDF并分块
pdf_path = "pdf_path.pdf"  # 替换为你的PDF路径
chunks = pdf_to_chunks(pdf_path, chunk_size=250, overlap_size=50)

# 生成嵌入向量
embeddings = generate_embeddings(chunks, model, tokenizer)

# 保存嵌入向量到文件
embeddings_path = "embeddings.pkl"
with open(embeddings_path, 'wb') as f:
    pickle.dump(embeddings, f)

# 创建FAISS索引并保存到文件
index_path = "faiss_index.bin"
index = create_faiss_index_and_save(embeddings, index_path)
# 查询最相似的块
def get_simlarity_chunks(query):
    results = query_similarity(index, query, model, tokenizer, chunks, merge_chunks=1)
    return results

# 保存chunks到文件
chunks_path = "chunks.pkl"
with open(chunks_path, 'wb') as f:
    pickle.dump(chunks, f)