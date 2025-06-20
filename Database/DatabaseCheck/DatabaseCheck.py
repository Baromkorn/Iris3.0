from pymilvus import MilvusClient, connections, utility

async def CheckDatabase() :
    client = MilvusClient(uri="http://localhost:19530")
    fmt = "\n=== {:30} ===\n"
    print(fmt.format("start connecting to Milvus"))
    connections.connect("default", host="localhost", port="19530")

    check = utility.has_collection("iris_collection")
    print(f"Does collection iris_collection exist in Milvus: {check}")
    return {"status": "success",
            "message": f"Does collection iris_collection exist in Milvus: {check}"}