import numpy as np
from pymilvus import MilvusClient, connections
from core.iris_setup import matcher
from utils.iris_utils import extract_template

async def VerifyUser_bothiris(cid: str, output_L, output_R):
    print("\n=== Connecting to Milvus ===")
    connections.connect("default", host="localhost", port="19530")
    client = MilvusClient(uri="http://localhost:19530")
    res = client.get_load_state(collection_name="iris_collection")
    print(res)
    if res.get("state") != "Loaded":
        client.load_collection(collection_name="iris_collection")
    data_L = output_L["iris_template"]
    data_R = output_R["iris_template"]

    # LEFT eye verification
    res_L = client.query(
        collection_name="iris_collection",
        filter=f'cid == "{cid}" and eye_side == "left"',
        output_fields=["iris_codes", "mask_codes"],
        limit=1
    )
    print("Left eye query result:", res_L)
    if not res_L:
        connections.disconnect("default")
        return {"status": "failed", "reason": "No matching left eye found in the database."}
    new_data_L = extract_template(res_L[0])
    distance_L = matcher.run(data_L, new_data_L)

    # RIGHT eye verification
    res_R = client.query(
        collection_name="iris_collection",
        filter=f'cid == "{cid}" and eye_side == "right"',
        output_fields=["iris_codes", "mask_codes"],
        limit=1
    )
    print("Right eye query result:", res_R)
    if not res_R:
        connections.disconnect("default")
        return {"status": "failed", "reason": "No matching right eye found in the database."}
    new_data_R = extract_template(res_R[0])
    distance_R = matcher.run(data_R, new_data_R)

    print("Distance of Left eye:", distance_L)
    print("Distance of Right eye:", distance_R)
    connections.disconnect("default")
    print("Disconnected from Milvus.")

    if distance_L <= 0.37 and distance_R <= 0.37:
        return {
            "status": "success",
            "match": True,
            "message": "Found a match in the database",
            "cid": cid
        }
    else:
        return {
            "status": "success",
            "match": False,
            "message": "No match found in database"
        }