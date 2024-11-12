import csv
import subprocess
from tqdm import tqdm  # 引入进度条库
import time
total_time = 0
query_count = 0

# 配置根目录和方法
root = './ragtest'
method = 'global' 

# 创建一个列表来存储带有答案的数据
output_data = []
max_questions = 20  # 设置最多读取的问题数量

# 读取CSV文件中的问题
with open('dataset.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    
    # 提取CSV文件的列名，并添加新的"answer"列
    fieldnames = reader.fieldnames + ['answer']

    # 使用tqdm为迭代添加进度条，限制只读取前5条
    for idx, row in tqdm(enumerate(reader), desc="Processing questions", total=max_questions, ncols=100):
        if idx >= max_questions:
            break  # 只处理前5个问题
        question = row['question'].strip()  # 读取并去掉多余空白符
        if question:
            # 构建命令
            start_time = time.time()  # 记录开始时间
            command = ['python', '-m', 'graphrag.query', '--root', root, '--method', method, question]
            
            # 执行命令并捕获输出，同时避免与进度条混合显示
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            end_time = time.time()  # 记录结束时间
            # 计算查询时间
            query_time = end_time - start_time
            total_time += query_time
            query_count += 1
            
            # 查找"SUCCESS: Global Search Response:"后的内容
            output = result.stdout
            if "SUCCESS: Global Search Response:" in output:
                response = output.split("SUCCESS: Global Search Response:")[1].strip()  # 提取后面的内容
                response = response.replace('\n', ' ')  # 移除换行符
                row['answer'] = response  # 将答案写入当前行的"answer"列
            else:
                row['answer'] = "未找到匹配的响应"
            
            # 将每一行带有答案的数据保存到列表
            output_data.append(row)
            
# 打印查询平均时间
print("查询平均时间：", total_time / query_count)

# 将带有答案的数据写入一个新的CSV文件
# with open('questions_with_answers.csv', 'w', newline='', encoding='utf-8') as csvfile:
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
#     # 写入表头
#     writer.writeheader()
    
#     # 写入每一行的数据
#     writer.writerows(output_data)

# print("问题和答案已保存到questions_with_answers.csv")
