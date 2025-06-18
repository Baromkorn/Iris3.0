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
async def SearchUser(output,eye_side,cid) :
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
        limit=100,
        filter=filter_term,
        output_fields=["pcode","cid","iris_codes","mask_codes"]
    )

    template_array = []
    pcode_array = []
    cid_array = []
    for hits in res:
        for hit in hits:
            vector = hit.entity.get("iris_codes")   # This is your stored binary vector
            mask = hit.entity.get("mask_codes")     # If stored
            pcode = hit.pcode                         # If you stored an "id" field
            cid = hit.cid
            pcode_array.append(pcode)
            cid_array.append(cid)
            mask = np.array(mask)
            mask = convert_bytes_to_bool_list(mask)
            masks = np.concatenate([
            code.flatten()
            for code in data.mask_codes
            ])
            mask = convert_bool_list_to_bytes(masks)
            combined_vector = vector+mask
            new_data = convert_bytes_to_template(combined_vector)
            template_array.append(new_data)

    print("Number of Entries: ",len(template_array))
    match_result = await Iris_Matcher(data,template_array)
    closest_match = match_result[0]
    closest_distance = match_result[1]
    index = match_result[2]
    
    

    print("Closest result has a distance of: "+ str(closest_distance))
    print("With pcode: ", pcode_array[index])
    print("cid: ", cid_array[index])
    connections.disconnect("default")
    print("Disconnected to Milvus.")
    if closest_distance > 0.37 :
        return ["No Match in system, HD>0.37"]
    if closest_distance <= 0.37 :
        return ["Matched with pcode: %s"%(pcode_array[index]),"and cid: %s"%(cid_array[index]),"Distance: %.6f"%(closest_distance)]