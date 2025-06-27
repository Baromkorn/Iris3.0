from pymilvus import connections, utility, MilvusClient
import numpy as np
from utils.file_utils import convert_bool_list_to_bytes

async def EnrollUser_singleiris(output, pcode, eye_side, cid):
    # Check for case with no pcode and cid
    if pcode == "" and cid == "" :
        return {"status": "failed", "reason": "No pcode and cid provided"}
    print("\n=== Connecting to Milvus ===\n")
    connections.connect("default", host="localhost", port="19530")
    client = MilvusClient(uri="http://localhost:19530")

    
    # Check if collection exists
    if not utility.has_collection("iris_collection"):
        connections.disconnect("default")
        return {"status": "failed", "reason": "Collection iris_collection does not exist."}

    # Extract and flatten iris and mask codes
    data = output["iris_template"]
    iris = np.concatenate([code.flatten() for code in data.iris_codes])
    mask = np.concatenate([code.flatten() for code in data.mask_codes])

    record = {
        "iris_codes": convert_bool_list_to_bytes(iris),
        "mask_codes": convert_bool_list_to_bytes(mask),
        "pcode": pcode,
        "cid": cid,
        "eye_side": eye_side
    }
    res = client.get_load_state(collection_name="iris_collection")
    print(res)
    if res.get("state") != "Loaded":
        client.load_collection(collection_name="iris_collection")
    # Check for existing user with same cid + eye_side
    existing = client.query(
        collection_name="iris_collection",
        filter=f'cid == "{cid}" and eye_side == "{eye_side}" and pcode == "{pcode}"',
        output_fields=["cid"],
        limit=1
    )

    
    if (existing and cid != "") or (existing and pcode != ""):
        connections.disconnect("default")
        return {"status": "failed", "reason": "Duplicate found in database"}

    # Insert record
    client.insert(data=[record], collection_name="iris_collection")

    # Load if not already loaded
    res = client.get_load_state(collection_name="iris_collection")
    print(res)
    if res.get("state") != "Loaded":
        client.load_collection(collection_name="iris_collection")

    connections.disconnect("default")
    print("Disconnected from Milvus.")

    return {"status": "success"}
