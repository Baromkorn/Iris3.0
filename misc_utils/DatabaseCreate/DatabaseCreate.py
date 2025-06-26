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

print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")

has = utility.has_collection("iris_collection")
print(f"Does collection iris_collection exist in Milvus: {has}")
if not has :
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="iris_codes", dtype=DataType.BINARY_VECTOR, dim=16384),
        FieldSchema(name="mask_codes",dtype=DataType.ARRAY, element_type=DataType.INT16, max_capacity=4096),
        FieldSchema(name="pcode", dtype=DataType.VARCHAR, max_length=13),
        FieldSchema(name="cid",dtype=DataType.VARCHAR,max_length=13),
        FieldSchema(name="eye_side",dtype=DataType.VARCHAR,max_length=5)
    ]

    schema = CollectionSchema(fields, description="Iris template storage")
    collection = Collection(name="iris_collection", schema=schema)

    print(fmt.format("Start Creating index IVF_FLAT"))
    index = {
        "index_type": "AUTOINDEX",
        "metric_type": "HAMMING"
    }

    collection.create_index("iris_codes", index)
connections.disconnect("default")
print("Disconnected to Milvus.")


