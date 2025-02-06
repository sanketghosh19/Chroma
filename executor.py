import os
from datetime import datetime
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

class ExcelLoader:
    """A simple loader for Excel files (XLSX or XLS)."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        # Read all sheets and convert them to Documents
        xls = pd.ExcelFile(self.file_path)
        docs = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            text = df.to_string(index=False)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": self.file_path,
                        "sheet_name": sheet_name
                    }
                )
            )
        return docs


class BRDRAG:
    def load_documents(self, document_paths: List[str]) -> List[Dict]:
        documents = []

        for path in document_paths:
            if path.lower().endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.lower().endswith((".docx", ".doc")):
                loader = Docx2txtLoader(path)
            elif path.lower().endswith((".xlsx", ".xls")):
                loader = ExcelLoader(path)
            else:
                print(f"Unsupported file type: {path}")
                continue

            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50
            )
            documents.extend(text_splitter.split_documents(docs))

        return documents

    def splitDoc(self, documents):

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=128
        )

        splits = text_splitter.split_documents(documents)
        return splits

    def getEmbedding(
        self,
    ):
        modelPath = "mixedbread-ai/mxbai-embed-large-v1"
        #device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        model_kwargs = {"device": device}  # cuda/cpu
        encode_kwargs = {"normalize_embeddings": False}

        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        
        #embeddings = FastEmbedEmbeddings()
        return embeddings


    def getResponse(self, document_paths: List[str], query: str) -> str:
        print("Loading documents...")
        documents = self.load_documents(document_paths)
        print("Documents loaded.")
        splits = self.splitDoc(documents)
        print("Documents split.")
        embeddings = self.getEmbedding()
        print("Embeddings loaded.")
        # get systen current tinme stamp
        current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        persist_directory = f"docs/chroma/{current_timestamp}"

        vectordb = Chroma.from_documents(
            documents=splits,  # splits we created earlier
            embedding=embeddings,
            persist_directory=persist_directory,  # save the directory
        )

        question = query
        docs = vectordb.similarity_search(question)
        return " ".join(doc.page_content for doc in docs)
