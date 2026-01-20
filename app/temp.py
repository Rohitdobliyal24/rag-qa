# # logger
# # config    yml file in production
# # core code 
# # api schemas   documents.py   health   query
# # main.py
#AWS IAM ECR app runner  create repo using cli 
# create dockerfile in code 
#docker ci/cd setup in notion
# .github workflow ci.yml  deploy.yml


# import tempfile
# from pathlib import Path
# from typing import BinaryIO

# from langchain_community.document_loaders import (
#     CSVLoader,
#     PyPDFLoader,
#     TextLoader,
# )
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from functools import lru_cache
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from typing import Any
# from uuid import uuid4
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.documents import Document
# from langchain_qdrant import QdrantVectorStore
# from qdrant_client import QdrantClient
# from qdrant_client.http.exceptions import UnexpectedResponse
# from qdrant_client.http.models import Distance, VectorParams
# from app.config import get_settings
# from app.utils.logger import get_logger
# settings = get_settings()

# logger = get_logger(__name__)
# class DocumentProcessor:
#     SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv"}
#     def __init__(
#      self,
#      chunk_size: int | None = None,
#      chunk_overlap: int | None = None 
#     ):
        
#      settings = get_settings()
#      self.chunk_size = chunk_size or settings.chunk_size
#      self.text_splitter = RecursiveCharacterTextSplitter(
#          chunk_size=self.chunk_size,
#          chunk_overlap=self.chunk_overlap,
         
#      )
#      logger.info(f"chunk overlap not found")
#     def load_pdf(self,file_path: str | Path)-> list[Document]:
#         file_path =Path(file_path)
#         loader =PyPDFLoader(str(file_path))
#         documents =loader.load()
#         return documents
#     def load_file(self,file_path:str| Path)-> list[Document]:
#         file_path =Path(file_path)
#         extension = file_path.suffix.lower()
#         if extension not in self.SUPPORTED_EXTENSIONS:
#             raise ValueError(
#                 f"Not found"
#             )
#         loaders = {
#             ".pdf": self.load_pdf,
#             ".txt": self.load_text,
#             ".csv": self.load_csv,
#         }
#         return loaders[extension](file_path)
#     def load_from_upload(
#         self,
#         file:BinaryIO,
#         filename:str,
#     ) -> list[Document]:
#         with tempfile.NamedTemporaryFile(
#             delete=False,
#             suffix="pdf",
#         ) as temp_file:
#             temp_file.write(file.read())
#             temp_path =temp_file.name
#         try:
#             documents = self.load_file(temp_path)
#             for doc in documents:
#                 doc.metadata["source"]=filename
#             return documents
#         finally:
#             Path(temp_path).unlink(missing_ok=True)
    
# @lru_cache
# def get_embeddings() -> GoogleGenerativeAIEmbeddings:
#     settings=get_settings()
#     embeddings=GoogleGenerativeAIEmbeddings(
#         model=settings.embedding_model,
#         google_api_key=settings.openai_api_key,
#     )
#     return embeddings

# class EmbeddingService:
#     def __init__(self):
#         settings =get_settings()
#         self.embeddings =get_embeddings()
        
#     def embed_query(self,text:str) -> list[float]:
#         return self.embeddings.embed_query(text)
#     def embed_document(self,text:list[str]) -> list[list[float]]:
#         return self.embeddings.aembed_documents(text)
        
    
# @lru_cache
# def get_qdrant_client()->QdrantClient:
#     client =QdrantClient(
#         url=settings.qdrant_url,
#         api_key=settings.qdrant_api_key,
#     )
#     return client

# class VectorStoreService:
#     def __init__(self, collection_name: str | None = None):
#         """Initialize vector store service.

#         Args:
#             collection_name: Name of the Qdrant collection (default from settings)
#         """
#         self.collection_name = collection_name or settings.collection_name
#         self.client = get_qdrant_client()
#         self.embeddings = get_embeddings()

#         # Ensure collection exists
#         self._ensure_collection()

#         # Initialize LangChain Qdrant vector store
#         self.vector_store = QdrantVectorStore(
#             client=self.client,
#             collection_name=self.collection_name,
#             embedding=self.embeddings,
#         )

#         logger.info(f"VectorStoreService initialized for collection: {self.collection_name}")
#     def _ensure_collection(self)->None:
#         try:
#             collection_info=self.client.get_collection(self.collection_name)
#         except UnexpectedResponse:
#             self.client.create_collection(
#                 collection_name=self.collection_name,
#                 vectors_config=VectorParams(
#                     size=1233,
#                     distance=Distance.COSINE
#                 )
#             )
#     def add_documents(self,documents:list[Document]) -> list[str]:
#         if not documents:
#             logger.warning('No documents')
#             return []
#         self.vector_store.add_documents(documents,ids=ids)
#     def search(self,query:str,k:int |None=None)->list[Document]:
#         k=5
#         result=self.vector_store.similarity_search(query,k=k)
#         return result
#     def delete_collection(self) -> None:
#         self.client.delete_collection(self.collection_name)
#     def get_collection_info(self) -> dict:
#         info=self.client.get_collection(self.collection_name)
#         return{
#             "point_count":info.points_count
#         }
        
# class RAGChain:
#     def __init__(self,vector_store_service:VectorStoreService | None = None):
#         self.vector_store =vector_store_service or VectorStoreService()
#         self.retriever=self.vector_store.get_retriever()
#         self._evaluator =None
#         self.llm = ChatGoogleGenerativeAI(
#             model=settings.llm_model,
#             temperature=settings.llm_temperature,
#             google_api_key=settings.google_api_key,
#         )
#         self.prompt = ChatPromptTemplate.from_template("ded")
#         self.chain =(
#             {
#                 "context":self.retriever,
#                 "question":RunnablePassthrough()
#             }
#             | self.prompt
#             | self.llm
#             | StrOutputParser()
#         )
       
            
#         def query(self,question:str) ->str:
#             try:
#                 answer =self.chain.invoke(question)
#                 source_doc =self.retriever.invoke(question)
#                 source =[
#                     "content":(
#                         doc.page_content[:500] +"...."
#                     ),
#                     "metadata":doc.metadata,
#                     for doc in source_doc
#                 ]
#                 return answer
#             except Exception as e:
#                 logger.error(f"Error")
        
