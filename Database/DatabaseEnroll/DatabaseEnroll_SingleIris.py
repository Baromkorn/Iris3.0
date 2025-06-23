from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    MilvusClient
)
import numpy as np
from fast_api.utils.file_utils import convert_bool_list_to_bytes

async def EnrollUser_singleiris(output,pcode,eye_side,cid) :
    
    client=MilvusClient(uri="http://localhost:19530")
    fmt = "\n=== {:30} ===\n"

    print(fmt.format("start connecting to Milvus"))
    connections.connect("default", host="localhost", port="19530")

    has = utility.has_collection("iris_collection")
    print(f"Does collection iris_collection exist in Milvus: {has}")
    data = output["iris_template"]
    combined_codes = np.concatenate([
        code.flatten()
        for code in data.iris_codes
    ])
    combined_masks = np.concatenate([
        code.flatten()
        for code in data.mask_codes
    ])

    Data = [
        {"iris_codes" : convert_bool_list_to_bytes(combined_codes),
         "mask_codes" : convert_bool_list_to_bytes(combined_masks),
        "pcode": pcode,
        "cid": cid,
        "eye_side": eye_side}
    ]

    existing = client.query(
        collection_name="iris_collection",
        filter=f'cid == "{cid}" and eye_side == "{eye_side}"',
        output_fields=["cid"],
        limit=1
    )
    connections.disconnect("default")
    print("Disconnected to Milvus.")
    if existing :
        return {"status": "failed",
                "reason": "duplicate found in database"}
    else:
        client.insert(data=Data ,collection_name="iris_collection")
        client.load_collection(collection_name="iris_collection")
        return {"status": "success"}