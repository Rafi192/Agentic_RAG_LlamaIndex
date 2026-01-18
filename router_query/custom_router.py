from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core import VectorStoreIndex, SummaryIndex, SimpleDirectoryReader
from llama_index.core.query_engine import CustomQueryEngine
#$loading my documents
documents = SimpleDirectoryReader('data').load_data()

#creating different query engines
vector_index = VectorStoreIndex.from_documents(documents)
vector_engine = vector_index.as_query_engine()

summary_index = SummaryIndex.from_documents(documents)
summary_engine = summary_index.as_query_engine()

from llama_index.core import SQLDatabase
from sqlalchemy import create_engine

sql_database = SQLDatabase.from_uri("my_database_uri")
sql_engine = sql_database.as_query_engine()



class router(CustomQueryEngine):

    def __init__(self, vector_engine, summary_engine, sql_engine):
        self.vector_engine = vector_engine
        self.summary_engine = summary_engine 
        self.sql_engine = sql_engine
    
    def custom_query(self, query_str : str):
        query_lower = query_str.lower()

        if any(word in query_lower for word in ["summary","overview","general"]):
            return self.summary_engine.query(query_str)
        
        if any( word in query_lower for word in ["count","sum","average", "total"]):
            return self.sql_engine(query_str)
        
        else:
            return self.vector_engine(query_str)

router_rr = router(vector_engine, summary_engine,sql_engine) # router parameters --
response = router_rr.query("Give me an overview of the document")           