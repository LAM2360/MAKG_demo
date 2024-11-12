import os
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from tqdm import tqdm
os.environ['OPENAI_API_KEY'] = "your openai api key"
os.environ['OPENAI_API_BASE'] = "your openai api base"

import pandas as pd
df = pd.read_csv("your dataset.csv")
# sample_df = df.sample(n=109, random_state=1)  # random_state for reproducibility
# print(sample_df)
# df = sample_df

llm = ChatOpenAI(model = 'gpt-4o', max_tokens=1000)

from langchain.prompts import ChatPromptTemplate
# Combine all contexts into a single string
#final_context = "\n\n".join(all_contexts)
#print(final_context)

# Create prompt template
PROMPT_TEMPLATE = """
你是一个专业机械故障诊断员，能够回答出问题{question}。请用中文回答，谢谢。
你只需要回答出答案即可，不用添加任何其他描述。
请一步一步思考。
"""
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

query= df['question']

responses = []
for q in tqdm(enumerate(query),desc='Processing Quesitons',total=len(query),ncols=100):
    prompt = prompt_template.format(question=q)
    model = llm 
    response_text = model.predict(prompt)
    answer = response_text.replace('\n','')
    answer = answer.replace('\n\n','')
    responses.append(answer)
    # print(f"Question : {q}")
    # print(f"Answer : {response_text}")

df['answer'] = responses

df.to_csv("dataset_gpt4o.csv", index=False)

print('done!')
