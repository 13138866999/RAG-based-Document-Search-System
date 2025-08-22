import os
from typing import List

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

Path_to_dataset = "../db"

def dataset_creation(chunks: List[Document], path: str = Path_to_dataset) -> None:
    embeddings = OpenAIEmbeddings()

    if os.path.exists(path):
        db = Chroma(persist_directory=path, embedding_function=embeddings)
        db.add_documents(chunks)
        db.persist()
        print("Dataset updated (appended).")
    else:
        db = Chroma.from_documents(chunks, embeddings, persist_directory=path)
        db.persist()
        print("Dataset created.")
