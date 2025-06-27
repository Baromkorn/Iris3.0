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
    template_array = []
    pcode_array = []
    cid_array = []
    v_distance_array = []
    matcher_inputs = []

    for hits in res:
        for hit in hits:
            vector = hit.entity.get("iris_codes")
            mask = hit.entity.get("mask_codes")
            pcode = hit.pcode
            cid = hit.cid

            # Process mask bytes
            mask = np.array(mask)
            mask = convert_bytes_to_bool_list(mask)
            masks = np.concatenate([code.flatten() for code in mask])
            mask_bytes = convert_bool_list_to_bytes(masks)

            combined_vector = vector + mask_bytes
            template = convert_bytes_to_template(combined_vector)

            matcher_inputs.append((template, pcode, cid))

    # Compute matcher scores
    results = []
    for i, (template, pcode, cid) in enumerate(matcher_inputs):
        match, hd, _ = await Iris_Matcher(data, [template])
        score = 2000 - int(hd * 2000)
        results.append({
            "pcode": pcode,
            "cid": cid,
            "hamming_distance": round(hd, 4),
            "score": score
        })

    # Sort results by lowest hamming distance
    results.sort(key=lambda x: x["hamming_distance"])

    # Filter by threshold (optional)
    #filtered_results = [r for r in results if r["hamming_distance"] <= 0.37]

    return results