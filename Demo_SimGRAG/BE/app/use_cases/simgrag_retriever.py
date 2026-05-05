import json
import re

class SimGRAGCypherRetriever:
    def __init__(self, milvus_adapter, neo4j_adapter):
        self.milvus = milvus_adapter
        self.neo4j = neo4j_adapter
        
    def parse_query_graph(self, llm_graph_output):
        try:
            # find JSON block in llm output
            match = re.search(r'\{.*\}', llm_graph_output, re.DOTALL)
            if match:
                data = match.group(0)
                # LLM output may have single quotes for tuples, let's fix it for python eval or json load
                try:
                    import ast
                    parsed = ast.literal_eval(data)
                    return parsed.get("graph", [])
                except:
                    pass
        except:
            pass
        return []
