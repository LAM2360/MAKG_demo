# pip install lightrag-hku

import os
import logging
import sys 
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from tqdm import tqdm
import time
import csv

start, end = 0, 0
total_time = 0

WORKING_DIR = "./your_working_dir"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    chunk_token_size = 3000,
    llm_model_name="",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
    ),
)

with open("./book_1.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

with open('your dataset.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)

    # 提取CSV文件的列名，并添加新的"answer"列
    fieldnames = reader.fieldnames + ['answer']
    max_questions = 210  # 设置最多读取的问题数量
    output_data = []
    # 使用tqdm为迭代添加进度条，限制只读取前5条
    for idx, row in tqdm(enumerate(reader), desc="Processing questions", total=max_questions, ncols=100):
        if idx >= max_questions:
            break
        question = row['question'].strip()
        if question:
            start = time.time()
            response = rag.query(question, param=QueryParam(mode="local"))
            end = time.time()
            query_time = end - start
            total_time += query_time
            row['answer'] = response
            # print(response)
        # 将带答案的列表保存为 outputdata
            output_data.append(row)
            
        
    print("查询平均时间：", total_time / max_questions)
    # 将带有答案的数据写入一个新的CSV文件，不存在就新建
with open("dataset_lightrag.csv",'w',encoding='utf-8',newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_data)

print("Done")
