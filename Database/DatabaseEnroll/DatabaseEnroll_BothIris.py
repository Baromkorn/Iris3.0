from pymilvus import connections, utility, MilvusClient
import numpy as np
from utils.file_utils import convert_bool_list_to_bytes

async def EnrollUser_bothiris(output_L, output_R, pcode, cid):
    print("\n=== Connecting to Milvus ===\n")
    connections.connect("default", host="localhost", port="19530")
    client = MilvusClient(uri="http://localhost:19530")

    # Check if collection exists
    if not utility.has_collection("iris_collection"):
        connections.disconnect("default")
        return {"status": "failed", "reason": "Collection iris_collection does not exist."}

    # Extract and flatten iris and mask codes
    def extract_data(output, side):
        iris = np.concatenate([code.flatten() for code in output["iris_template"].iris_codes])
        mask = np.concatenate([code.flatten() for code in output["iris_template"].mask_codes])
        return {
            "iris_codes": convert_bool_list_to_bytes(iris),
            "mask_codes": convert_bool_list_to_bytes(mask),
            "pcode": pcode,
            "cid": cid,
            "eye_side": side
        }

    data_L = extract_data(output_L, "left")
    data_R = extract_data(output_R, "right")

    res = client.get_load_state(collection_name="iris_collection")
    print(res)
    if res.get("state") != "Loaded":
        client.load_collection(collection_name="iris_collection")
    # Check for existing user
    existing = client.query(
        collection_name="iris_collection",
        filter=f'cid == "{cid}"',
        output_fields=["cid"],
        limit=1
    )
    
    if existing:
        connections.disconnect("default")
        return {"status": "failed", "reason": "Duplicate found in database"}

    # Insert both eyes
    client.insert(data=[data_L], collection_name="iris_collection")
    client.insert(data=[data_R], collection_name="iris_collection")

    # Load collection if not already loaded
    res = client.get_load_state(collection_name="iris_collection")
    print(res)
    if res.get("state") != "Loaded":
        client.load_collection(collection_name="iris_collection")
    connections.disconnect("default")
    print("Disconnected from Milvus.")

    return {"status": "success"}