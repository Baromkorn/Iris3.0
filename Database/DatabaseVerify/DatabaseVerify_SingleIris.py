import numpy as np
from pymilvus import (
    MilvusClient,
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from fast_api.utils.file_utils import convert_bool_list_to_bytes, convert_bytes_to_template, convert_bytes_to_bool_list
from fast_api.core.iris_setup import matcher

async def VerifyUser_singleiris (cid: str, output, eye_side) :
    client = MilvusClient(uri="http://localhost:19530")
    fmt = "\n=== {:30} ===\n"
    print(fmt.format("start connecting to Milvus"))
    connections.connect("default", host="localhost", port="19530")

    data = output["iris_template"]
    res = client.query(
        collection_name="iris_collection",
        filter=f'cid == "{cid}" and eye_side == "{eye_side}"',
        output_fields=["iris_codes","mask_codes"],
        limit=1
    )
    print(f"{eye_side} eye query result:", res)
    if not res:
        return {"status": "failed", "reason": "No matching cid found in the database."}
    for hit in res:
        vector = hit.get("iris_codes")   # This is your stored binary vector
        mask = hit.get("mask_codes")
        mask = convert_bytes_to_bool_list(mask)
        masks = np.concatenate([
        code.flatten()
        for code in mask])
        mask = convert_bool_list_to_bytes(masks)
        vector = vector[0]
        combined_vector = vector+mask
        new_data = convert_bytes_to_template(combined_vector)     # If stored
    distance = matcher.run(data, new_data)
    print("Distance of Left eye", distance)
    connections.disconnect("default")
    print("Disconnected to Milvus.")
    if distance <= 0.37:
        return {"status": "success",
                "match": True,
                "message": "found a match in the database",
                "cid": cid}
    else :
        return {"status": "success",
                "match": False,
                "message": "no match found in database"}
