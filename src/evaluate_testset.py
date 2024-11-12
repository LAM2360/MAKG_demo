import requests
import json
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import load_from_disk
import pandas as pd
from datasets import Dataset
from ragas.metrics import (
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
)
from ragas import evaluate
from dotenv import load_dotenv
from tqdm import tqdm

df = pd.read_csv('your testset.csv')
df['contexts'] = df['contexts'].apply(lambda x: x.split(',') if isinstance(x, str) else x)
print(df.columns)
# df.drop(columns=['retrival_contexts'], inplace=True)
print(df.columns)
dataset = Dataset.from_pandas(df)

os.environ['OPENAI_API_KEY'] = "your openai api key"
os.environ['OPENAI_API_BASE'] = "your openai api base"
llm = ChatOpenAI(model="gpt-4")

result = evaluate(
    dataset,
    metrics=[
        context_precision,
        faithfulness,
        context_recall,
        answer_correctness,
    ],
    llm=llm,
)
print(result)
# result.to_pandas().to_csv("./result.csv")
