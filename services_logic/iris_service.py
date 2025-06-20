import numpy as np
import cv2
from fastapi import UploadFile, HTTPException, File
from typing import Optional
from core.iris_setup import iris_pipeline, matcher
from utils.file_utils import save_file, load_image
from Database.DatabaseEnroll.DatabaseEnroll import EnrollUser_singleiris, EnrollUser_bothiris
from Database.DatabaseSearch.DatabaseSearch import SearchUser_singleiris, SearchUser_bothiris
from Database.DatabaseVerify.DatabaseVerify import VerifyUser

async def enroll_user(
    left_iris: Optional[UploadFile] = File(None),
    right_iris: Optional[UploadFile] = File(None),
    pcode: str = "",
    cid: str = ""
):
    if not left_iris and not right_iris:
        raise HTTPException(status_code=400, detail="At least one iris image must be provided.")

    output_L = None
    output_R = None

    if left_iris:
        try:
            left = await left_iris.read()
            left_img = cv2.imdecode(np.frombuffer(left, np.uint8), cv2.IMREAD_GRAYSCALE)
            output_L = iris_pipeline(img_data=left_img, eye_side="left")
            if output_L["error"] is not None:
                raise ValueError("Left eye image error")
        except Exception:
            output_L = None

    if right_iris:
        try:
            right = await right_iris.read()
            right_img = cv2.imdecode(np.frombuffer(right, np.uint8), cv2.IMREAD_GRAYSCALE)
            output_R = iris_pipeline(img_data=right_img, eye_side="right")
            if output_R["error"] is not None:
                raise ValueError("Right eye image error")
        except Exception:
            output_R = None

    if not output_L and not output_R:
        raise HTTPException(status_code=400, detail="Neither image contains a valid iris.")

    # Call your enroll logic based on what is available
    if output_L and output_R:
        status = await EnrollUser_bothiris(output_L=output_L, output_R=output_R, pcode=pcode, cid=cid)
    elif output_L:
        status = await EnrollUser_singleiris(output=output_L, eye_side="left", pcode=pcode, cid=cid)
    elif output_R:
        status = await EnrollUser_singleiris(output=output_R, eye_side="right", pcode=pcode, cid=cid)

    return status

async def verify_user(cid: str, left_iris: Optional[UploadFile] = File(None), right_iris: Optional[UploadFile]= File(None)):
    if not left_iris and not right_iris:
        raise HTTPException(status_code=400, detail="At least one iris image must be provided.")

    output_L = None
    output_R = None

    if left_iris:
        try:
            left = await left_iris.read()
            left_img = cv2.imdecode(np.frombuffer(left, np.uint8), cv2.IMREAD_GRAYSCALE)
            output_L = iris_pipeline(img_data=left_img, eye_side="left")
            if output_L["error"] is not None:
                raise ValueError("Left eye image error")
        except Exception:
            output_L = None

    if right_iris:
        try:
            right = await right_iris.read()
            right_img = cv2.imdecode(np.frombuffer(right, np.uint8), cv2.IMREAD_GRAYSCALE)
            output_R = iris_pipeline(img_data=right_img, eye_side="right")
            if output_R["error"] is not None:
                raise ValueError("Right eye image error")
        except Exception:
            output_R = None

    if not output_L and not output_R:
        raise HTTPException(status_code=400, detail="Neither image contains a valid iris.")
    status = await VerifyUser(output_L=output_L, output_R=output_R, cid=cid)
    return status

async def search_user(left_iris: Optional[UploadFile] = File(None), right_iris: Optional[UploadFile] = File(None)) :

    output_L = None
    output_R = None

    if left_iris:
        try:
            left = await left_iris.read()
            left_img = cv2.imdecode(np.frombuffer(left, np.uint8), cv2.IMREAD_GRAYSCALE)
            output_L = iris_pipeline(img_data=left_img, eye_side="left")
            if output_L["error"] is not None:
                raise ValueError("Left eye image error")
        except Exception:
            output_L = None

    if right_iris:
        try:
            right = await right_iris.read()
            right_img = cv2.imdecode(np.frombuffer(right, np.uint8), cv2.IMREAD_GRAYSCALE)
            output_R = iris_pipeline(img_data=right_img, eye_side="right")
            if output_R["error"] is not None:
                raise ValueError("Right eye image error")
        except Exception:
            output_R = None

    if not output_L and not output_R:
        raise HTTPException(status_code=400, detail="Neither image contains a valid iris.")

    # Call your search logic based on what is available
    if output_L and output_R:
        status = await SearchUser_bothiris(output_L=output_L, output_R=output_R)
    elif output_L:
        status = await SearchUser_singleiris(output=output_L, eye_side="left")
    elif output_R:
        status = await SearchUser_singleiris(output=output_R, eye_side="right")
    
    return status