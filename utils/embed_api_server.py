from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
from fastapi import FastAPI
import torch
from transformers import AutoTokenizer

app = FastAPI()
model_path = "your model path"
# 加载模型和分词器
model = SentenceTransformer(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


class Item(BaseModel):
    input: list
    model: str
    encoding_format: str = None


@app.post("/v1/embeddings")
async def create_embedding(item: Item):
    # 确保输入是字符串列表
    texts = [str(x) for x in item.input]

    # 计算token数量
    tokens = tokenizer(texts, padding=True, truncation=True)
    token_count = sum(len(ids) for ids in tokens['input_ids'])

    # 生成嵌入
    with torch.no_grad():
        embeddings = model.encode(texts, convert_to_tensor=True)

    # 将张量转换为列表
    embeddings_list = embeddings.tolist()

    # 构建响应
    data = [
        {
            "object": "embedding",
            "index": i,
            "embedding": emb
        }
        for i, emb in enumerate(embeddings_list)
    ]

    return {
        "object": "list",
        "data": data,
        "model": "acge_text_embedding",  # 改回原来的模型名称
        "usage": {
            "prompt_tokens": token_count,
            "total_tokens": token_count
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8166)