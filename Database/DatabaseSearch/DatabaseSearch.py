import numpy as np
from pymilvus import (
    MilvusClient,
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from utils.file_utils import convert_bool_list_to_bytes
async def SearchUser(output,eye_side,cid) :
    client = MilvusClient(uri="http://localhost:19530")
    fmt = "\n=== {:30} ===\n"

    print(fmt.format("start connecting to Milvus"))
    connections.connect("default", host="localhost", port="19530")
    data = output["iris_template"]
    client.load_collection(collection_name="iris_collection")
    subject_combined = np.concatenate([
        code.flatten()
        for code in data.iris_codes + data.mask_codes
    ]).tolist()
    search_params = {
        "params": {"nprobe": 10},
        "metric_type": "HAMMING"
    }

    query_vector = convert_bool_list_to_bytes(subject_combined)
    res = client.search(
        collection_name="iris_collection",
        data=[query_vector],
        anns_field="iris_vector",
        search_params=search_params,
        limit=5,
        output_fields=["pcode"]
    )

    print(res)
    return res