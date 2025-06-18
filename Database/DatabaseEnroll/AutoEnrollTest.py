import os
import iris
import numpy as np
import cv2
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    MilvusClient
)
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
workdir = r'C:\Users\USER\Desktop\open-iris-main\BMT-20'
print(workdir)
client=MilvusClient(uri="http://localhost:19530")
connections.connect("default", host="localhost", port="19530")
for sets in os.listdir(workdir):
    temp_path1 = os.path.join(workdir,sets)
    for side in os.listdir(temp_path1):
        temp_path2 = os.path.join(temp_path1,side)
        for pic in os.listdir(temp_path2):
            temp_pic = cv2.imread(os.path.join(temp_path2,pic), cv2.IMREAD_GRAYSCALE)
            if side == 'L':
                eye_side = "left"
            #elif side == 'R':
                #eye_side = "right"
                count = pic[9:13]
                output = ir_pipeline(img_data=temp_pic, eye_side=eye_side)
                data = output["iris_template"]
                if output["error"] is not None:
                    continue
                combined_codes = np.concatenate([
                code.flatten()
                for code in data.iris_codes
                ])
                combined_masks = np.concatenate([
                code.flatten()
                for code in data.mask_codes
                ])
                Data = [
                {"iris_codes" : convert_bool_list_to_bytes(combined_codes),
                "mask_codes" : convert_bool_list_to_bytes(combined_masks),
                "pcode": "p"+str(count),
                "cid": str(sets),
                "eye_side":eye_side}
                ]
                client.insert(data=Data ,collection_name="iris_collection")
                client.load_collection(collection_name="iris_collection")