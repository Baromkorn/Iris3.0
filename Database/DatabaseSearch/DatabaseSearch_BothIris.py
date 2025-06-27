import numpy as np
from pymilvus import MilvusClient, connections
from utils.file_utils import convert_bool_list_to_bytes, convert_bytes_to_template, convert_bytes_to_bool_list
from utils.iris_utils import process_search_results

async def SearchUser_bothiris(output_L, output_R):
    client = MilvusClient(uri="http://localhost:19530")
    print("\n=== Start connecting to Milvus ===\n")
    connections.connect("default", host="localhost", port="19530")

    data_L = output_L["iris_template"]
    data_R = output_R["iris_template"]

    res = client.get_load_state(collection_name="iris_collection")
    print(res)
    if res.get("state") != "Loaded":
        client.load_collection(collection_name="iris_collection")

    combined_codes_L = np.concatenate([code.flatten() for code in data_L.iris_codes]).tolist()
    combined_codes_R = np.concatenate([code.flatten() for code in data_R.iris_codes]).tolist()

    search_params = {
        "params": {"nprobe": 10},
        "metric_type": "HAMMING"
    }

    # Left eye search
    query_vector_L = convert_bool_list_to_bytes(combined_codes_L)
    res_L = client.search(
        collection_name="iris_collection",
        data=[query_vector_L],
        anns_field="iris_codes",
        search_params=search_params,
        limit=5,
        filter='eye_side like "left%"',
        output_fields=["pcode", "cid", "iris_codes", "mask_codes"]
    )
    left_results = await process_search_results(res_L, data_L, "left")

    # Right eye search
    query_vector_R = convert_bool_list_to_bytes(combined_codes_R)
    res_R = client.search(
        collection_name="iris_collection",
        data=[query_vector_R],
        anns_field="iris_codes",
        search_params=search_params,
        limit=5,
        filter='eye_side like "right%"',
        output_fields=["pcode", "cid", "iris_codes", "mask_codes"]
    )
    right_results = await process_search_results(res_R, data_R, "right")

    connections.disconnect("default")
    print("Disconnected from Milvus.")

    # Merge left and right eye results
    combined_rank = left_results + right_results

    # Deduplicate by keeping only highest score for each (pcode, cid)
    unique_results = {}
    for entry in combined_rank:
        key = (entry["pcode"], entry["cid"])
        if key not in unique_results or entry["score"] > unique_results[key]["score"]:
            unique_results[key] = entry

    # Convert to list and get top 5 by score
    filtered_top5 = sorted(unique_results.values(), key=lambda x: x["score"], reverse=True)[:5]

    if not filtered_top5:
        return {"status": "failed", "reason": "No matching person across both eyes"}

    return {
        "status": "success",
        "results": filtered_top5
    }
