import numpy as np
from pymilvus import (
    MilvusClient,
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from utils.file_utils import convert_bool_list_to_bytes, convert_bytes_to_template, convert_bytes_to_bool_list
from core.iris_setup import matcher

async def VerifyUser_bothiris (cid: str, output_L, output_R) :
    client = MilvusClient(uri="http://localhost:19530")

    fmt = "\n=== {:30} ===\n"

    print(fmt.format("start connecting to Milvus"))
    connections.connect("default", host="localhost", port="19530")
    # Check for left side match
    data_L = output_L["iris_template"]
    res_L = client.query(
        collection_name="iris_collection",
        filter=f'cid == "{cid}" and eye_side == "left"',
        output_fields=["iris_codes","mask_codes"],
        limit=1
    )
    print("Left eye query result:", res_L)
    if not res_L :
        return {"status": "failed", "reason": "No matching cid found in the database."}
    for hit in res_L:
        vector = hit.get("iris_codes")  
        mask = hit.get("mask_codes")
        mask = np.array(mask)
        mask = convert_bytes_to_bool_list(mask)
        masks = np.concatenate([
        code.flatten()
        for code in mask])
        mask = convert_bool_list_to_bytes(masks)
        vector = vector[0]
        combined_vector = vector+mask
        new_data_L = convert_bytes_to_template(combined_vector)     # If stored
    distance_L = matcher.run(data_L, new_data_L)
    # Check for right side match
    data_R = output_R["iris_template"]
    res_R = client.query(
        collection_name="iris_collection",
        filter=f'cid == "{cid}" and eye_side == "right"',
        output_fields=["iris_codes","mask_codes"],
        limit=1
    )
    print("Right eye query result:", res_R)
    if not res_R:
        return {"status": "failed", "reason": "No matching cid found in the database."}
    for hit in res_R:
        vector = hit.get("iris_codes")   # This is your stored binary vector
        mask = hit.get("mask_codes")
        mask = convert_bytes_to_bool_list(mask)
        masks = np.concatenate([
        code.flatten()
        for code in mask])
        mask = convert_bool_list_to_bytes(masks)
        vector = vector[0]
        combined_vector = vector+mask
        new_data_R = convert_bytes_to_template(combined_vector)     # If stored
    distance_R = matcher.run(data_R, new_data_R)
    print("Distance of Left eye", distance_L)
    print("Distance of Right eye", distance_R)
    connections.disconnect("default")
    print("Disconnected to Milvus.")
    if distance_L <= 0.37 and distance_R <= 0.37 :
        return {"status": "success",
                "match": True,
                "message": "found a match in the database",
                "cid": cid}
    else :
        return {"status": "success",
                "match": False,
                "message": "no match found in database"}


