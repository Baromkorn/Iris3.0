import numpy as np
import os
import cv2
import time
import json
from pymilvus import (
    MilvusClient,
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
from fast_api.utils.file_utils import convert_bool_list_to_bytes, convert_bytes_to_template, convert_bytes_to_bool_list
from fast_api.utils.iris_utils import Iris_Matcher
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
bad_pic = list()
client = MilvusClient(uri="http://localhost:19530")
fmt = "\n=== {:30} ===\n"
print(fmt.format("start connecting to Milvus"))
connections.connect("default", host="localhost", port="19530")
workdir = r'C:\Users\User\OneDrive\Desktop\fast_api_folder\fast_api\BMT-20'
workdir_new = r'C:\Users\User\OneDrive\Desktop\fast_api_folder\fast_api\Dataset_NECTEC_25072023'
p1 = r'C:\Users\User\OneDrive\Desktop\fast_api_folder\fast_api\BMT-20\1\L\BMT20L_BI0002_1.bmp'
p2 = r'C:\Users\User\OneDrive\Desktop\fast_api_folder\fast_api\BMT-20\1\R\BMT20R_BI0002_1.bmp'

#p1 = p1.replace('L','R')
#p1[-21] = 'R'
#p1[-14] = 'R'
iris_data_folder = "iris_dat"
count_fp = 0
count = ''
total_run = 0
fail = 0
for sets in os.listdir(workdir_new):
    if sets=='1':
        continue
    if sets.startswith('.'):
        continue
    temp_path1 = os.path.join(workdir_new,sets,iris_data_folder)
    for side in os.listdir(temp_path1):
        if side.endswith(".txt"):
            continue
        if side == "EYE_R.bmp":
            continue
        temp_path2_L = os.path.join(temp_path1,side)
        temp_pic_L = cv2.imread(temp_path2_L, cv2.IMREAD_GRAYSCALE)
        temp_path2_R = temp_path2_L.replace('L','R')
        if not os.path.exists(temp_path2_R):
            fail+=1
            continue
        temp_pic_R = cv2.imread(temp_path2_R, cv2.IMREAD_GRAYSCALE)
        output_L = iris_pipeline(img_data=temp_pic_L, eye_side="left")
        output_R = iris_pipeline(img_data=temp_pic_R, eye_side="right")
        #count = pic[9:13]
        if output_L["error"] is not None or output_R["error"] is not None:
            continue
        total_run +=1
        template_L = output_L["iris_template"]
        template_R = output_R["iris_template"]
        client.load_collection(collection_name="iris_collection")
        combined_codes_L = np.concatenate([
            code.flatten()
            for code in template_L.iris_codes
        ]).tolist()
        combined_codes_R = np.concatenate([
            code.flatten()
            for code in template_R.iris_codes
        ]).tolist()
        search_params = {
            "params": {"nprobe": 10},
            "metric_type": "HAMMING"
        }
        query_vector_L = convert_bool_list_to_bytes(combined_codes_L)
        query_vector_R = convert_bool_list_to_bytes(combined_codes_R)
        filter_term_L = 'eye_side like "left"'
        filter_term_R = 'eye_side like "right"'
        res_L = client.search(
            collection_name="iris_collection",
            data=[query_vector_L],
            anns_field="iris_codes",
            search_params=search_params,
            limit=5,
            filter=filter_term_L,
            output_fields=["pcode","cid","iris_codes","mask_codes"]
        )
        res_R = client.search(
            collection_name="iris_collection",
            data=[query_vector_R],
            anns_field="iris_codes",
            search_params=search_params,
            limit=5,
            filter=filter_term_R,
            output_fields=["pcode","cid","iris_codes","mask_codes"]
        )
        template_array_L = []
        pcode_array_L = []
        cid_array_L = []
        v_distance_array_L = []
        for hits in res_L:
            for hit in hits:
                vector_L = hit.entity.get("iris_codes")   # This is your stored binary vector
                mask_L = hit.entity.get("mask_codes")     # If stored
                pcode_L = hit.pcode                         # If you stored an "id" field
                cid_L = hit.cid
                v_distance_L = hit.distance
                pcode_array_L.append(pcode_L)
                cid_array_L.append(cid_L)
                v_distance_array_L.append(v_distance_L)
                mask_L = np.array(mask_L)
                mask_L = convert_bytes_to_bool_list(mask_L)
                masks_L = np.concatenate([
                code.flatten() for code in mask_L
                ])
                mask_L = convert_bool_list_to_bytes(masks_L)
                combined_vector_L = vector_L+mask_L
                new_data_L = convert_bytes_to_template(combined_vector_L)
                template_array_L.append(new_data_L)
        template_array_R = []
        pcode_array_R = []
        cid_array_R = []
        v_distance_array_R = []
        for hits in res_R:
            for hit in hits:
                vector_R = hit.entity.get("iris_codes")   # This is your stored binary vector
                mask_R = hit.entity.get("mask_codes")     # If stored
                pcode_R = hit.pcode                         # If you stored an "id" field
                cid_R = hit.cid
                v_distance_R = hit.distance
                pcode_array_R.append(pcode_R)
                cid_array_R.append(cid_R)
                v_distance_array_R.append(v_distance_R)
                mask_R = np.array(mask_R)
                mask_R = convert_bytes_to_bool_list(mask_R)
                masks_R = np.concatenate([
                code.flatten() for code in mask_R
                ])
                mask_R = convert_bool_list_to_bytes(masks_R)
                combined_vector_R = vector_R+mask_R
                new_data_R = convert_bytes_to_template(combined_vector_R)
                template_array_R.append(new_data_R)
        match_result_L = top5(template_L,template_array_L)
        match_result_R = top5(template_R,template_array_R)

        closest_match_L = match_result_L[0][2]
        closest_distance_L = match_result_L[0][0]
        index_L = match_result_L[0][1]
        closest_match_R = match_result_R[0][2]
        closest_distance_R = match_result_R[0][0]
        index_R = match_result_R[0][1]
        fp_check = False
        print(sets)
        print('Match_result_L:',closest_distance_L,index_L)
        print('Match_result_R:',closest_distance_R,index_R)
        if match_result_L[0][3] or match_result_R[0][3]:
            print("Count=",count)
            print("pcode=",pcode_array_L[index_L][1:5])
            print("No Match in system, HD>0.37")
            print(closest_distance_L)
        if not (match_result_L[0][3] or match_result_R[0][3]) :
            print("Count=",count)
            print("pcode=",pcode_array_L[index_L][1:5])
            print("Matched with pcode: %s"%(pcode_array_L[index_L]),"and cid: %s"%(cid_array_L[index_L]),"Distance: %.6f"%(closest_distance_L))
            print("pcode_array_L[index_L]: ",pcode_array_L[index_L],"pcode_array_R[index_R]",pcode_array_R[index_R])
            print("cid_array_L[index_L]:",cid_array_L[index_L],"cid_array_R[index_R]:",cid_array_R[index_R])
            print("count:",count,"pcode_array_L[index_L][1:5]:",pcode_array_L[index_L][1:5])
            fp_check = True
        print(fp_check)
        if fp_check:
            count_fp+=1
        print("Count_fp=",count_fp)
        print("Total run=",total_run)
        print("-------------------")
connections.disconnect("default")
print("Disconnected to Milvus.")
print("Count_fn=",count_fp)
print("Total run=",total_run)
print("Fail=",fail)
print(count_fp/total_run*100)