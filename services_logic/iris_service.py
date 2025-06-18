import numpy as np
import cv2
from fastapi import UploadFile, HTTPException
from core.iris_setup import iris_pipeline, matcher
from utils.file_utils import save_file, load_image
from Database.DatabaseEnroll.DatabaseEnroll import EnrollUser_singleiris, EnrollUser_bothiris
from Database.DatabaseSearch.DatabaseSearch import SearchUser_singleiris, SearchUser_bothiris
async def enroll_user_singleiris(file: UploadFile, pcode: str, eye_side: str, cid: str):
    content = await file.read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE)

    output = iris_pipeline(img_data=img, eye_side=eye_side)
    if output["error"] is not None:
        raise HTTPException(status_code=400, detail="Image does not contain iris")
    status = await EnrollUser_singleiris(output=output,pcode=pcode,eye_side=eye_side,cid=cid)
    return status

async def enroll_user_bothiris(left_iris: UploadFile,right_iris: UploadFile, pcode: str, cid: str):
    left = await left_iris.read()
    right = await right_iris.read()
    left_img = cv2.imdecode(np.frombuffer(left, np.uint8), cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imdecode(np.frombuffer(right, np.uint8), cv2.IMREAD_GRAYSCALE)

    output_L = iris_pipeline(img_data=left_img, eye_side="left")
    output_R = iris_pipeline(img_data=right_img, eye_side="right")
    if output_L["error"] is not None or output_R["error"] is not None:
        raise HTTPException(status_code=400, detail="Image does not contain iris")
    status = await EnrollUser_bothiris(output_L=output_L,output_R=output_R,pcode=pcode,cid=cid)
    return status

async def verify_user(file: UploadFile, pcode: str, eye_side: str):
    content = await file.read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE)

    output = iris_pipeline(img_data=img, eye_side=eye_side)
    if output["error"] is not None:
        raise HTTPException(status_code=400, detail="Image does not contain iris")

    stored_img = load_image(f"{pcode}_{eye_side}.png")
    output2 = iris_pipeline(img_data=stored_img, eye_side=eye_side)

    distance = matcher.run(output2["iris_template"], output["iris_template"])
    return {"Verification": str(distance <= 0.37), "distance": distance}

async def search_user_singleiris(file: UploadFile, eye_side: str):
    content = await file.read()
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE)

    output = iris_pipeline(img_data=img, eye_side=eye_side)
    if output["error"] is not None:
        raise HTTPException(status_code=400, detail="Image does not contain iris")
    status = await SearchUser_singleiris(output=output,eye_side=eye_side)
    return status

async def search_user_bothiris(left_iris: UploadFile,right_iris: UploadFile) :
    left = await left_iris.read()
    right = await right_iris.read()
    left_img = cv2.imdecode(np.frombuffer(left, np.uint8), cv2.IMREAD_GRAYSCALE)
    right_img = cv2.imdecode(np.frombuffer(right, np.uint8), cv2.IMREAD_GRAYSCALE)
    output_L = iris_pipeline(img_data=left_img, eye_side="left")
    output_R = iris_pipeline(img_data=right_img, eye_side="right")

    if output_L["error"] is not None or output_R["error"] is not None:
        raise HTTPException(status_code=400, detail="Image does not contain iris")
    status = await SearchUser_bothiris(output_L=output_L,output_R=output_R)
    return status