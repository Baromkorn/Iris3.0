import numpy as np
from pymilvus import (
    MilvusClient,
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from utils.file_utils import convert_bool_list_to_bytes, convert_bytes_to_template, convert_bytes_to_bool_list
from utils.iris_utils import process_search_results  # Make sure to import

async def SearchUser_singleiris(output, eye_side):
    client = MilvusClient(uri="http://localhost:19530")
    print("\n=== Start connecting to Milvus ===\n")
    connections.connect("default", host="localhost", port="19530")

    data = output["iris_template"]
    res = client.get_load_state(collection_name="iris_collection")
    print(res)
    if res.get("state") != "Loaded":
        client.load_collection(collection_name="iris_collection")

    combined_codes = np.concatenate([
        code.flatten()
        for code in data.iris_codes
    ]).tolist()

    search_params = {
        "params": {"nprobe": 10},
        "metric_type": "HAMMING"
    }

    query_vector = convert_bool_list_to_bytes(combined_codes)
    filter_term = f'eye_side like "{eye_side}%"'

    res = client.search(
        collection_name="iris_collection",
        data=[query_vector],
        anns_field="iris_codes",
        search_params=search_params,
        limit=5,
        filter=filter_term,
        output_fields=["pcode", "cid", "iris_codes", "mask_codes"]
    )

    top_matches = await process_search_results(res, data, eye_side)
    connections.disconnect("default")
    print("Disconnected from Milvus.")

    if not top_matches:
        return {
            "status": "failed",
            "reason": "No valid matches in system"
        }

    return {
        "status": "success",
        "results": top_matches
    }