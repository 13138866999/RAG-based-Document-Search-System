from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def read_single_file(path: str):
    _, file_extension = os.path.splitext(path)

    loader = None

    match file_extension.lower():
        case ".pdf":
            loader = PyMuPDFLoader(path)
        case ".txt":
            loader = TextLoader(path, encoding='utf-8')
        case ".docx":
            loader = UnstructuredWordDocumentLoader(path, mode="single")
        case _:
            print(f"Unsupported file type: {file_extension}")
            return []

    return loader.load()

def read_full_dir(path: str):
    loader_mapping = {
        ".pdf": PyMuPDFLoader,
        ".txt": TextLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".doc": UnstructuredWordDocumentLoader,
    }

    all_docs = []
    for extension, loader_cls in loader_mapping.items():
        print(f"Loading files with extension: {extension}")

        kwargs = {}
        if extension == ".txt":
            kwargs = {'encoding': 'utf-8'}

        loader = DirectoryLoader(
            path,
            glob=f"**/*{extension}",
            loader_cls=loader_cls,
            loader_kwargs=kwargs,
            show_progress=True,
            use_multithreading=True
        )

        docs = loader.load()
        if docs:
            all_docs.extend(docs)

    return all_docs

def loader_md (path: str):
    loader = DirectoryLoader(path, glob="**/*.md")
    doc = loader.load()
    return doc

def split_to_chunk(doc):
    text_spliter =  RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap= 500, add_start_index=True, length_function=len)
    chunks = text_spliter.split_documents(doc)
    return chunks