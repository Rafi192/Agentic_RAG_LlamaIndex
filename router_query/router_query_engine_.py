from llama_index.core import VectorStoreIndex, SummaryIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core import Settings
from llama_index.core import StorageContext, load_index_from_storage
from dotenv import load_dotenv
import logging
import os

load_dotenv()
logger = logging.getLogger(__name__)

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Set up Gemini LLM and Embeddings
llm_gemini = Gemini(model="models/gemini-2.0-flash-exp", api_key=os.getenv("GEMINI_API_KEY"))
embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=os.getenv("GEMINI_API_KEY"))

# Configure global settings
Settings.llm = llm_gemini
Settings.embed_model = embed_model

# Loading docs
# if os.path.exists()
doc_2024 = SimpleDirectoryReader(r"data").load_data()
doc_2008 = SimpleDirectoryReader(r"data_two").load_data()

if doc_2024:
    logger.info("directory is read")
else:
    logger.error("could not load documents")

# Creating indexes (now using Gemini embeddings)
try:
    index_pdf = VectorStoreIndex.from_documents(doc_2024)
    index_2008 = VectorStoreIndex.from_documents(doc_2008)
    if index_pdf and index_2008:
        logger.info("Vector index created!")
except Exception as e:
    logger.error(f"could not create a vector index: {e}")

# Creating the query engine
engine_one = index_pdf.as_query_engine(similarity_top_k=3)
engine_two = index_2008.as_query_engine(similarity_top_k=3)

# Defining tools with description
query_engine_tools = [
    QueryEngineTool(
        query_engine=engine_one,
        metadata={
            "name": "financial_2024",
            "description": "contains financial data from 2024 documents including Q3 2023 revenue"
        }
    ),
    QueryEngineTool(
        query_engine=engine_two,
        metadata={
            "name": "financial_2008",
            "description": "Financial crisis of 2008 data and analysis"
        }
    )
]

# Creating router with LLM selector
router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(llm=llm_gemini),
    query_engine_tools=query_engine_tools,
    verbose=True
)

response = router_query_engine.query("what was the revenue in Q3 2023?")
print(response)