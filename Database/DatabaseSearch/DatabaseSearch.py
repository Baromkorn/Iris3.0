import numpy as np
from pymilvus import (
    MilvusClient,
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from utils.file_utils import convert_bool_list_to_bytes, convert_bytes_to_template, convert_bytes_to_bool_list
from utils.iris_utils import Iris_Matcher
async def SearchUser_singleiris(output,eye_side) :
    client = MilvusClient(uri="http://localhost:19530")
    fmt = "\n=== {:30} ===\n"

    print(fmt.format("start connecting to Milvus"))
    connections.connect("default", host="localhost", port="19530")
    data = output["iris_template"]
    client.load_collection(collection_name="iris_collection")
    combined_codes = np.concatenate([
        code.flatten()
        for code in data.iris_codes
    ]).tolist()
    search_params = {
        "params": {"nprobe": 10},
        "metric_type": "HAMMING"
    }

    query_vector = convert_bool_list_to_bytes(combined_codes)
    filter_term = 'eye_side like "%s%%"'%eye_side
    res = client.search(
        collection_name="iris_collection",
        data=[query_vector],
        anns_field="iris_codes",
        search_params=search_params,
        limit=5,
        filter=filter_term,
        output_fields=["pcode","cid","iris_codes","mask_codes"]
    )

    template_array = []
    pcode_array = []
    cid_array = []
    v_distance_array = []
    for hits in res:
        for hit in hits:
            vector = hit.entity.get("iris_codes")   # This is your stored binary vector
            mask = hit.entity.get("mask_codes")     # If stored
            pcode = hit.pcode                         # If you stored an "id" field
            cid = hit.cid
            v_distance = hit.distance
            pcode_array.append(pcode)
            cid_array.append(cid)
            v_distance_array.append(v_distance)
            mask = np.array(mask)
            mask = convert_bytes_to_bool_list(mask)
            masks = np.concatenate([
            code.flatten() for code in mask
            ])
            mask = convert_bool_list_to_bytes(masks)
            combined_vector = vector+mask
            new_data = convert_bytes_to_template(combined_vector)
            template_array.append(new_data)

    print("Top 10 pcode: ",pcode_array[0:11])
    print("Top 10 cid: ",cid_array[0:11])
    print("Top 10 distance: ",v_distance_array[0:11])
    print("Number of Entries: ",len(template_array))
    match_result = await Iris_Matcher(data,template_array)
    closest_match = match_result[0]
    closest_distance = match_result[1]
    index = match_result[2]
    
    hd = closest_distance
    scaled_hd = int(hd * 2000)
    print("Closest result has a distance of: "+ str(closest_distance))
    print("With pcode: ", pcode_array[index])
    print("cid: ", cid_array[index])
    connections.disconnect("default")
    print("Disconnected to Milvus.")
    if closest_distance > 0.37 :
        return {"status": "failed",
                "reason": "No Match in system, HD>0.37"}
    if closest_distance <= 0.37 :
        return {"pcode": pcode_array[index],"cid": (cid_array[index]), "score: ": (scaled_hd)}
    
async def SearchUser_bothiris(output_L, output_R) :
    client = MilvusClient(uri="http://localhost:19530")
    fmt = "\n=== {:30} ===\n"
    print(fmt.format("start connecting to Milvus"))
    connections.connect("default", host="localhost", port="19530")

    data_L = output_L["iris_template"]
    data_R = output_R["iris_template"]
    client.load_collection(collection_name="iris_collection")
    combined_codes_L = np.concatenate([
        code.flatten()
        for code in data_L.iris_codes
    ]).tolist()
    combined_codes_R = np.concatenate([
        code.flatten()
        for code in data_R.iris_codes
    ]).tolist()
    search_params = {
        "params": {"nprobe": 10},
        "metric_type": "HAMMING"
    }

    query_vector_L = convert_bool_list_to_bytes(combined_codes_L)
    res = client.search(
        collection_name="iris_collection",
        data=[query_vector_L],
        anns_field="iris_codes",
        search_params=search_params,
        limit=5,
        filter='eye_side like "left%"',
        output_fields=["pcode","cid","iris_codes","mask_codes"]
    )

    template_array_L = []
    pcode_array_L = []
    cid_array_L = []
    v_distance_array_L = []
    for hits in res:
        for hit in hits:
            vector = hit.entity.get("iris_codes")   # This is your stored binary vector
            mask = hit.entity.get("mask_codes")     # If stored
            pcode = hit.pcode                         
            cid = hit.cid
            v_distance = hit.distance
            pcode_array_L.append(pcode)
            cid_array_L.append(cid)
            v_distance_array_L.append(v_distance)
            mask = np.array(mask)
            mask = convert_bytes_to_bool_list(mask)
            masks = np.concatenate([
            code.flatten()
            for code in mask])
            mask = convert_bool_list_to_bytes(masks)
            combined_vector = vector+mask
            new_data_L = convert_bytes_to_template(combined_vector)
            template_array_L.append(new_data_L)

    print("Top 10 pcode_L: ",pcode_array_L[0:11])
    print("Top 10 cid_L: ",cid_array_L[0:11])
    print("Top 10 distance_L: ",v_distance_array_L[0:11])
    print("Number of Entries for Left: ",len(template_array_L))
    match_result_L = await Iris_Matcher(data_L,template_array_L)
    closest_match_L = match_result_L[0]
    closest_distance_L = match_result_L[1]
    index_L = match_result_L[2]
    
    search_params = {
        "params": {"nprobe": 10},
        "metric_type": "HAMMING"
    }

    query_vector_R = convert_bool_list_to_bytes(combined_codes_R)
    res = client.search(
        collection_name="iris_collection",
        data=[query_vector_R],
        anns_field="iris_codes",
        search_params=search_params,
        limit=5,
        filter='eye_side like "right%"',
        output_fields=["pcode","cid","iris_codes","mask_codes"]
    )

    template_array_R = []
    pcode_array_R = []
    cid_array_R = []
    v_distance_array_R = []
    for hits in res:
        for hit in hits:
            vector = hit.entity.get("iris_codes")   # This is your stored binary vector
            mask = hit.entity.get("mask_codes")     # If stored
            pcode = hit.pcode                         
            cid = hit.cid
            v_distance = hit.distance
            pcode_array_R.append(pcode)
            cid_array_R.append(cid)
            v_distance_array_R.append(v_distance)
            mask = np.array(mask)
            mask = convert_bytes_to_bool_list(mask)
            masks = np.concatenate([
            code.flatten()
            for code in mask])
            mask = convert_bool_list_to_bytes(masks)
            combined_vector = vector+mask
            new_data_R = convert_bytes_to_template(combined_vector)
            template_array_R.append(new_data_R)

    print("Top 10 pcode_R: ",pcode_array_R[0:11])
    print("Top 10 cid_R: ",cid_array_R[0:11])
    print("Top 10 distance_R: ",v_distance_array_R[0:11])
    print("Number of Entries for Right: ",len(template_array_R))
    match_result_R = await Iris_Matcher(data_R,template_array_R)
    closest_match_R = match_result_R[0]
    closest_distance_R = match_result_R[1]
    index_R = match_result_R[2]
    hd = (closest_distance_L+closest_distance_R)/2
    scaled_hd = int(hd * 2000)
    connections.disconnect("default")
    print("Disconnected to Milvus.")
    if pcode_array_L[index_L] == pcode_array_R[index_R] and cid_array_L[index_L] == cid_array_R[index_R] :
        if closest_distance_L > 0.37 and closest_distance_R > 0.37:
            return {"status": "failed",
                    "reason": "No Match in system, HD>0.37"}
        if closest_distance_L <= 0.37 and closest_distance_R <= 0.37 :
            return {"status": "success",
                    "pcode": pcode_array_L[index_L],
                    "cid": (cid_array_L[index_L]),
                    "score: ": (scaled_hd)
                    }
    else :
        return{"status": "failed",
               "reason": "Left and Right Iris do not match"}