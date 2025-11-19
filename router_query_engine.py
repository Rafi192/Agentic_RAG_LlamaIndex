# from helper import get_openai_api_key
# OPENAI_API_KEY = get_openai_api_key()

import nest_asyncio
import os

nest_asyncio.apply()

print("Applied nest_asyncio to allow nested event loops.", flush=True)

#loading data 

from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"Loaded OPENAI_API_KEY from .env: {OPENAI_API_KEY is not None}", flush=True)


documents = SimpleDirectoryReader(input_files=[r"C:\Users\hasan\Rafi_SAA\practice_project_1\Agentic_RAG_LlamaIndex\data\meta_gpt.pdf"]).load_data()

print(f"Loaded {len(documents)} documents.", flush=True)


#defining the LLM and Embedding model
from openai import OpenAI

from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)

nodes = splitter.get_nodes_from_documents(documents)
print(f"Split documents into {len(nodes)} nodes.", flush=True)

#LLM
Settings.llm = LlamaOpenAI(
    model="Qwen/Qwen3-4B-Instruct-2507:nscale",
    temperature=0,
    max_tokens=1024,
    openai_api_key=os.environ["HF_TOKEN_READ"],
)
print("Configured LLM with Qwen3-4B-Instruct-2507:nscale model.", flush=True)
# print(f"LLM Settings: {Settings.llm}", flush=True)
#Embedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Configured embedding model using HuggingFaceEmbedding.", flush=True)


#defingin summary index and vector index over the same data
from llama_index.core import SummaryIndex, VectorStoreIndex
summary_index = SummaryIndex(nodes)
print("Summary Index created.", flush=True)
print( " summary index", summary_index, flush=True)
# print("Summary index length:", len(summary_index), flush=True)
vector_index = VectorStoreIndex(nodes)
print("Vector Index created.", flush=True)
# print("Vector index length:", len(vector_index), flush=True)



# these codes are out of date after llama index update
# service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
# print("Defined LLM Predictor and Service Context.", flush=True)
# print(f"LLM Predictor: {llm_predictor}", flush=True)
# print(f"Service Context: {service_context}", flush=True)

