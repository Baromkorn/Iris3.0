from pymilvus import MilvusClient, connections
from typing import Union

async def check_available(pcode_or_cid: str) -> dict:
    """
    Check whether a given pcode or cid is already in the database.

    Args:
        pcode_or_cid (str): A string that starts with 'p' followed by digits (pcode) or just digits (cid).

    Returns:
        dict: {
            "status": "success" or "failed",
            "available": bool (only if status is success),
            "reason": str (only if status is failed)
        }
    """

    # Validate input
    if not pcode_or_cid:
        return {"status": "failed", "reason": "Empty value"}

    if pcode_or_cid.startswith("p") and pcode_or_cid[1:].isdigit():
        filter_expr = f'pcode == "{pcode_or_cid}"'
    elif pcode_or_cid.isdigit():
        filter_expr = f'cid == "{pcode_or_cid}"'
    else:
        return {"status": "failed", "reason": "Invalid value format"}

    print("\n=== Connecting to Milvus ===\n")
    connections.connect("default", host="localhost", port="19530")
    client = MilvusClient(uri="http://localhost:19530")

    res = client.get_load_state(collection_name="iris_collection")
    print(res)
    if res.get("state") != "Loaded":
        client.load_collection(collection_name="iris_collection")

    try:
        res = client.query(
            collection_name="iris_collection",
            filter=filter_expr,
            output_fields=["pcode", "cid"],
            limit=1
        )
    except Exception as e:
        connections.disconnect("default")
        print("Disconnected from Milvus.")
        return {"status": "failed", "reason": str(e)}

    connections.disconnect("default")
    print("Disconnected from Milvus.")

    return {
        "status": "success",
        "available": bool(res)  # True if found, False otherwise
    }