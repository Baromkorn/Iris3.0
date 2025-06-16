from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    MilvusClient
)
import cv2

client = MilvusClient(uri="http://localhost:19530")

fmt = "\n=== {:30} ===\n"
search_latency_fmt = "search latency = {:.4f}s"
num_entities, dim = 3000, 32768

print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

has = utility.has_collection("iris_collection")
print(f"Does collection iris_collection exist in Milvus: {has}")
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="iris_vector", dtype=DataType.BINARY_VECTOR, dim=32768),
    FieldSchema(name="pcode", dtype=DataType.VARCHAR, max_length=13),
    FieldSchema(name="cid",dtype=DataType.VARCHAR,max_length=13)
]

schema = CollectionSchema(fields, description="Iris template storage")
collection = Collection(name="iris_collection", schema=schema)

print(fmt.format("Start Creating index IVF_FLAT"))
index = {
    "index_type": "AUTOINDEX",
    "metric_type": "HAMMING"
}

collection.create_index("iris_vector", index)

