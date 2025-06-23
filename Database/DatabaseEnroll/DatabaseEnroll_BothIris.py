from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    MilvusClient
)
import numpy as np
from fast_api.utils.file_utils import convert_bool_list_to_bytes
async def EnrollUser_bothiris(output_L,output_R,pcode,cid) :
    client=MilvusClient(uri="http://localhost:19530")
    fmt = "\n=== {:30} ===\n"
    print(fmt.format("start connecting to Milvus"))
    connections.connect("default", host="localhost", port="19530")

    has = utility.has_collection("iris_collection")
    print(f"Does collection iris_collection exist in Milvus: {has}")
    data_L = output_L["iris_template"]
    data_R = output_R["iris_template"]
    combined_codes_L = np.concatenate([
        code.flatten()
        for code in data_L.iris_codes
    ])
    combined_masks_L = np.concatenate([
        code.flatten()
        for code in data_L.mask_codes
    ])
    combined_codes_R = np.concatenate([
        code.flatten()
        for code in data_R.iris_codes
    ])
    combined_masks_R = np.concatenate([
        code.flatten()
        for code in data_R.mask_codes
    ])

    Data_L = [
        {"iris_codes" : convert_bool_list_to_bytes(combined_codes_L),
         "mask_codes" : convert_bool_list_to_bytes(combined_masks_L),
        "pcode": pcode,
        "cid": cid,
        "eye_side": "left"}
    ]
    Data_R = [
        {"iris_codes" : convert_bool_list_to_bytes(combined_codes_R),
         "mask_codes" : convert_bool_list_to_bytes(combined_masks_R),
        "pcode": pcode,
        "cid": cid,
        "eye_side": "right"}
    ]
    existing = client.query(
        collection_name="iris_collection",
        filter=f'cid == "{cid}"',
        output_fields=["cid"],
        limit=1
    )
    connections.disconnect("default")
    print("Disconnected to Milvus.")
    if existing :
        return {"status": "failed",
                "reason": "duplicate found in database"}
    else :
        client.insert(data=Data_L ,collection_name="iris_collection")
        client.insert(data=Data_R ,collection_name="iris_collection")
        client.load_collection(collection_name="iris_collection")
        return {"status": "success"}