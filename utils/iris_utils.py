from core.iris_setup import  matcher
import time
import numpy as np
from utils.file_utils import convert_bool_list_to_bytes, convert_bytes_to_template, convert_bytes_to_bool_list

async def Iris_Matcher(probe_template,gallery_array) :
    start = time.time()
    lowest_HD = 1.0
    i = 0
    closest_template = None
    for gallery_template in gallery_array :
        hamming_distance = matcher.run(probe_template, gallery_template)
        if lowest_HD > hamming_distance :
            lowest_HD = hamming_distance
            closest_template = gallery_template
            index = i
        i+=1
    end = time.time()
    print(f"Execution time of Matcher: {end - start:.4f} seconds")
    return closest_template,lowest_HD,index

def extract_template(hit: dict) -> bytes:
    """Reconstruct the iris template from iris and mask codes."""
    vector = hit["iris_codes"][0]
    raw_mask = np.array(hit["mask_codes"])
    bool_mask = np.concatenate([code.flatten() for code in convert_bytes_to_bool_list(raw_mask)])
    combined_vector = vector + convert_bool_list_to_bytes(bool_mask)
    return convert_bytes_to_template(combined_vector)

async def process_search_results(res, data, eye_side):
    """Helper to process Milvus search results and compute matcher distances."""
    template_array = []
    pcode_array = []
    cid_array = []
    v_distance_array = []

    for hits in res:
        for hit in hits:
            vector = hit.entity.get("iris_codes")   # stored binary vector
            mask = hit.entity.get("mask_codes")     # stored mask
            pcode = hit.pcode
            cid = hit.cid
            v_distance = hit.distance

            pcode_array.append(pcode)
            cid_array.append(cid)
            v_distance_array.append(v_distance)

            # process mask bytes
            mask = np.array(mask)
            mask = convert_bytes_to_bool_list(mask)
            masks = np.concatenate([code.flatten() for code in mask])
            mask = convert_bool_list_to_bytes(masks)

            combined_vector = vector + mask
            new_data = convert_bytes_to_template(combined_vector)
            template_array.append(new_data)

    # Run matcher and get closest result info
    match_result = await Iris_Matcher(data, template_array)
    closest_match, closest_distance, index = match_result

    return {
        "pcode_array": pcode_array,
        "cid_array": cid_array,
        "v_distance_array": v_distance_array,
        "closest_match": closest_match,
        "closest_distance": closest_distance,
        "index": index
    }