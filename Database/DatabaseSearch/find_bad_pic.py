import os
import iris
import numpy as np
import cv2
import json
def convert_bool_list_to_bytes(bool_list):
    if len(bool_list) % 8 != 0:
        raise ValueError("The length of a boolean list must be a multiple of 8")

    byte_array = bytearray(len(bool_list) // 8)
    for i, bit in enumerate(bool_list):
        if bit == 1:
            index = i // 8
            shift = i % 8
            byte_array[index] |= (1 << shift)
    return bytes(byte_array)
ir_pipeline = iris.IRISPipeline()
#use your own directory
workdir = r'C:\Users\User\OneDrive\Desktop\fast_api_folder\fast_api\BMT-20'
print(workdir)
badpic = dict()
for sets in os.listdir(workdir):
    temp_path1 = os.path.join(workdir,sets)
    for side in os.listdir(temp_path1):
        temp_path2 = os.path.join(temp_path1,side)
        for pic in os.listdir(temp_path2):
            temp_pic = cv2.imread(os.path.join(temp_path2,pic), cv2.IMREAD_GRAYSCALE)
            if side == 'L':
                eye_side = "left"
            elif side == 'R':
                eye_side = "right"
            count = pic[9:13]
            output = ir_pipeline(img_data=temp_pic, eye_side=eye_side)
            data = output["iris_template"]
            if output["error"] is not None:
                badpic[pic] = {"error_details": output["error"],"eye_side":eye_side}
                continue
with open("data.json", "w") as json_file:
    json.dump(badpic, json_file, indent=4)