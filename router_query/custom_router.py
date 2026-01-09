from llama_index.core.query_engine import CustomQueryEngine

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

router_rr = router(vector_engine, summary_engine,sql_engine)
response = router_rr.query("Give me an overview of the document")
        