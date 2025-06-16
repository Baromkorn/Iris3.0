from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    MilvusClient
)
import numpy as np
from utils.file_utils import convert_bool_list_to_bytes
async def EnrollUser(output,pcode,eye_side,cid) :
    
    client=MilvusClient(uri="http://localhost:19530")
    fmt = "\n=== {:30} ===\n"
    search_latency_fmt = "search latency = {:.4f}s"
    num_entities, dim = 3000, 32768

    print(fmt.format("start connecting to Milvus"))
    connections.connect("default", host="localhost", port="19530")

    has = utility.has_collection("iris_collection")
    print(f"Does collection iris_collection exist in Milvus: {has}")
    data = output["iris_template"]
    combined_vector = np.concatenate([
        code.flatten()
        for code in data.iris_codes + data.mask_codes
    ])

    Data = [
        {"iris_vector" : convert_bool_list_to_bytes(combined_vector),
        "pcode": pcode,
        "cid": cid}
    ]
    client.insert(data=Data ,collection_name="iris_collection")
    client.load_collection(collection_name="iris_collection")
    return {"status": "success"}