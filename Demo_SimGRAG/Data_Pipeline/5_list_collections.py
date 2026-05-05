from pymilvus import connections
connections.connect("default", host="localhost", port="19530")
from pymilvus import MilvusClient
client = MilvusClient("http://localhost:19530")
collections = client.list_collections()
print("Collections in Milvus:")
for c in collections:
    print(f"  - {c}")
