import numpy as np
from pymilvus import MilvusClient, connections
from core.iris_setup import matcher
from utils.iris_utils import extract_template


async def VerifyUser_singleiris(cid: str, output, eye_side: str):
    print("\n=== Connecting to Milvus ===")
    connections.connect("default", host="localhost", port="19530")
    client = MilvusClient(uri="http://localhost:19530")
    res = client.get_load_state(collection_name="iris_collection")
    print(res)
    if res.get("state") != "Loaded":
        client.load_collection(collection_name="iris_collection")
    data = output["iris_template"]

    res = client.query(
        collection_name="iris_collection",
        filter=f'cid == "{cid}" and eye_side == "{eye_side}"',
        output_fields=["iris_codes", "mask_codes"],
        limit=1
    )
    print(f"{eye_side.capitalize()} eye query result:", res)

    if not res:
        connections.disconnect("default")
        return {
            "status": "failed",
            "reason": f"No matching {eye_side} eye for cid {cid} found in the database."
        }

    new_data = extract_template(res[0])
    distance = matcher.run(data, new_data)
    print(f"Distance of {eye_side} eye:", distance)

    connections.disconnect("default")
    print("Disconnected from Milvus.")

    if distance <= 0.37:
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
            "message": "No match found in the database"
        }
