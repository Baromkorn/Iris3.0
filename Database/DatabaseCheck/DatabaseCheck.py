from pymilvus import MilvusClient, connections, utility

async def check_available(pcode_or_cid: str) -> bool:
    """
    Check that given pcode_or_cid is available or not
    """
    if pcode_or_cid :
        if pcode_or_cid.startswith("p") and pcode_or_cid[1:].isdigit():
            filtered = f'pcode == "{pcode_or_cid}"'
        elif pcode_or_cid.isdigit():
            filtered = f'cid == "{pcode_or_cid}"'
        else:
            return {"status": "failed",
                    "reason": "Invalid value"}
    else :
        return {"status": "failed",
                "reason": "Empty value"}
    client = MilvusClient(uri="http://localhost:19530")
    fmt = "\n=== {:30} ===\n"
    print(fmt.format("start connecting to Milvus"))
    client.load_collection("iris_collection")
    res = client.query(collection_name="iris_collection",
            filter=filtered, 
            output_fields=["pcode", "cid"])
    connections.disconnect("default")
    print("Disconnected to Milvus.")
    if res :
        return {"status": "success",
                "available": True}
    else :
        return {"status": "success",
                "available": False}