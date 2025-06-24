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

    # Search Left eye
    query_vector_L = convert_bool_list_to_bytes(combined_codes_L)
    res_L = client.search(
        collection_name="iris_collection",
        data=[query_vector_L],
        anns_field="iris_codes",
        search_params=search_params,
        limit=5,
        filter='eye_side like "left%"',
        output_fields=["pcode","cid","iris_codes","mask_codes"]
    )
    left_results = await process_search_results(res_L, data_L, "left")

    # Search Right eye
    query_vector_R = convert_bool_list_to_bytes(combined_codes_R)
    res_R = client.search(
        collection_name="iris_collection",
        data=[query_vector_R],
        anns_field="iris_codes",
        search_params=search_params,
        limit=5,
        filter='eye_side like "right%"',
        output_fields=["pcode","cid","iris_codes","mask_codes"]
    )
    right_results = await process_search_results(res_R, data_R, "right")

    print(f"Top 5 pcode_L: {left_results['pcode_array'][:5]}")
    print(f"Top 5 pcode_R: {right_results['pcode_array'][:5]}")

    # Defensive: check if no matches found
    if not left_results['pcode_array'] or not right_results['pcode_array']:
        connections.disconnect("default")
        return {"status": "failed", "reason": "No search results returned for one or both eyes"}

    # Average Hamming distance
    hd = (left_results['closest_distance'] + right_results['closest_distance']) / 2
    scaled_hd = int(hd * 2000)
    score = 2000-scaled_hd

    # Disconnect
    connections.disconnect("default")
    print("Disconnected from Milvus.")

    idx_L = left_results['index']
    idx_R = right_results['index']

    # Check matching pcode and cid
    if (left_results['pcode_array'][idx_L] == right_results['pcode_array'][idx_R] and
        left_results['cid_array'][idx_L] == right_results['cid_array'][idx_R]):

        if left_results['closest_distance'] > 0.37 or right_results['closest_distance'] > 0.37:
            return {"status": "failed", "reason": "No Match in system, HD > 0.37"}

        return {
            "status": "success",
            "pcode": left_results['pcode_array'][idx_L],
            "cid": left_results['cid_array'][idx_L],
            "score": score
        }
    else:
        return {"status": "failed", "reason": "Left and Right Iris do not match"}
