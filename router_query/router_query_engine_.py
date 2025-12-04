from llama_index.core import VectorStoreIndex, SummaryIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.openai import OpenAI

#embedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


#loading documents
docs_exm_1 = SimpleDirectoryReader(r"../data/data1", required_exts=[".pdf"]).load_data()
docs_exm_2 = SimpleDirectoryReader(r"../data/data2", required_exts=[".pdf"]).load_data()
print(len(docs_exm_1))
print(len(docs_exm_2))

#Creating indexes
indexes_data1 = VectorStoreIndex.from_documents(docs_exm_1)
indexes_data2 = VectorStoreIndex.from_documents(docs_exm_2)


print("data vector index",indexes_data1)
print("type of the vector--" ,type(indexes_data1))
print("data index -2 ",indexes_data2)