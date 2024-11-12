
# -*- coding: utf-8 -*-
# @file: make_ft_corpus.py
import os
from llama_index.legacy.finetuning import (
    generate_qa_embedding_pairs
)
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
# from dotenv import load_dotenv

# load_dotenv()

project_dir = os.path.dirname(os.path.abspath(__file__)).split('/src')[0]

TRAIN_FILES = [os.path.join(project_dir, "data/ft_train.txt")]
VAL_FILES = [os.path.join(project_dir, "data/ft_test.txt")]

TRAIN_CORPUS_FPATH = os.path.join(project_dir, "data/ft_train_corpus.json")
VAL_CORPUS_FPATH = os.path.join(project_dir, "data/ft_val_corpus.json")


def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter(chunk_size=250, chunk_overlap=0)
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes


train_nodes = load_corpus(TRAIN_FILES, verbose=True)
val_nodes = load_corpus(VAL_FILES, verbose=True)

api_key = "your openai api key"
api_base= "your openai api base"
llm = OpenAI(api_key=api_key,base_url = api_base)

qa_generate_prompt_tmpl = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a professor. Your task is to create {num_questions_per_chunk} questions for an upcoming quiz/examination based on the provided mechanical fault diagnosis-related text. The questions should cover diverse aspects of the content, ensuring a variety of question types without repetition. The questions should focus on mechanical fault diagnosis, without options, and should not start with "Q1" or "Q2". They should be closely aligned with the provided context, aiming to assess the students’ understanding of mechanical fault diagnosis theory and practice.
"""

train_dataset = generate_qa_embedding_pairs(nodes=train_nodes, llm=llm, num_questions_per_chunk=1, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl)
val_dataset = generate_qa_embedding_pairs(nodes=val_nodes, llm=llm, num_questions_per_chunk=1, qa_generate_prompt_tmpl=qa_generate_prompt_tmpl)

train_dataset.save_json(TRAIN_CORPUS_FPATH)
val_dataset.save_json(VAL_CORPUS_FPATH)