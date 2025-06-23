import numpy as np
import os
import cv2
import time
from pymilvus import (
    MilvusClient,
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from fast_api.utils.file_utils import convert_bool_list_to_bytes, convert_bytes_to_template, convert_bytes_to_bool_list
from fast_api.utils.iris_utils import Iris_Matcher
from fast_api.services_logic.iris_service import search_user_bothiris
from fast_api.core.iris_setup import iris_pipeline,matcher
from fast_api.Database.DatabaseSearch.DatabaseSearch_BothIris import SearchUser_bothiris
def top5(probe_template,gallery_array) :
    start = time.time()
    dist_list = []
    i = 0
    for gallery_template in gallery_array :
        hamming_distance = matcher.run(probe_template, gallery_template)
        dist_list.append([hamming_distance,i,gallery_template,hamming_distance>0.37])
        #dist_list.append([hamming_distance,i,hamming_distance>0.37])
        i+=1
    dist_list.sort()
    end = time.time()
    print(f"Execution time of Matcher: {end - start:.4f} seconds")
    return dist_list
columns = "ID,Eye_side,Rank1,Dis,Rank2,Dis,Rank3,Dis,Rank4,Dis,Rank5,Dis"
answer = list()
client = MilvusClient(uri="http://localhost:19530")
fmt = "\n=== {:30} ===\n"
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")
workdir = r'C:\Users\User\OneDrive\Desktop\fast_api_folder\fast_api\BMT-20'
workdir_new = r'C:\Users\User\OneDrive\Desktop\fast_api_folder\fast_api\Dataset_NECTEC_25072023'
'''
img1 = r'C:\\Users\\User\\OneDrive\\Desktop\\fast_api_folder\\fast_api\\BMT-20\\2\\L\\BMT20L_BI0003_2.bmp'
img2 = r'C:\\Users\\User\\OneDrive\\Desktop\\fast_api_folder\\fast_api\\BMT-20\\1\\L\\BMT20L_BI0004_1.bmp'
temp1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
temp2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
print(1)
output1 = iris_pipeline(img_data=temp1, eye_side="left")
output2 = iris_pipeline(img_data=temp2, eye_side="left")
o1 = output1["iris_template"]
o2 = output2["iris_template"]
print(top5(o1,[o2]))'''
iris_data_folder = "iris_dat"
count_fp = 0
count_fn = 0
for sets in os.listdir(workdir_new):
    if sets=='1':
        continue
    if sets.startswith('.'):
        continue
    temp_path1 = os.path.join(workdir_new,sets,iris_data_folder)
    for side in os.listdir(temp_path1):
        if side.endswith(".txt"):
            continue
        temp_path2 = os.path.join(temp_path1,side)
        #for pic in os.listdir(temp_path2):
        temp_pic = cv2.imread(temp_path2, cv2.IMREAD_GRAYSCALE)
        if side == 'EYE_L.bmp':
            eye_side = "left"
        elif side == 'EYE_R.bmp':
            eye_side = "right"
        #count = pic[9:13]
        output = iris_pipeline(img_data=temp_pic, eye_side=eye_side)
        if output["error"] is not None:
            continue
        template = output["iris_template"]

        
        client.load_collection(collection_name="iris_collection")
        combined_codes = np.concatenate([
            code.flatten()
            for code in template.iris_codes
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

        match_result = top5(template,template_array)
        print(sets,eye_side)
        fp_check = False
        temp_ans = list()
        temp_ans.append(sets)
        temp_ans.append(eye_side)
        for i in range(5):
            closest_match = match_result[i][2]
            closest_distance = match_result[i][0]
            index = match_result[i][1]
            temp_ans.append(pcode_array[index])
            temp_ans.append(closest_distance)
            if match_result[i][3] :
                print("No Match in system, HD>0.37")
                print(closest_distance)
            if not match_result[i][3] :
                fp_check = True
                print("Matched with pcode: %s"%(pcode_array[index]),"and cid: %s"%(cid_array[index]),"Distance: %.6f"%(closest_distance))
        answer.append(temp_ans)
        print(fp_check)
        if fp_check:
            count_fp+=1
        print("-------------------")
connections.disconnect("default")
print("Disconnected to Milvus.")
print(count_fp/1034*100)
data = np.array(answer, dtype=object)
np.savetxt("single_eye_fpir.csv", data, fmt='%s', delimiter=",", header=columns, comments='')