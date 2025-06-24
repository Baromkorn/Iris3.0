import numpy as np
import cv2
from fastapi import UploadFile, HTTPException, File
from typing import Optional, Tuple
from core.iris_setup import iris_pipeline
from Database.DatabaseEnroll.DatabaseEnroll_SingleIris import EnrollUser_singleiris
from Database.DatabaseEnroll.DatabaseEnroll_BothIris import EnrollUser_bothiris
from Database.DatabaseSearch.DatabaseSearch_BothIris import SearchUser_bothiris
from Database.DatabaseSearch.DatabaseSearch_SingleIris import SearchUser_singleiris
from Database.DatabaseVerify.DatabaseVerify_BothIris import VerifyUser_bothiris
from Database.DatabaseVerify.DatabaseVerify_SingleIris import VerifyUser_singleiris
from utils.SaveImage import save_iris_image
import uuid


# Helper function to process an iris image
async def process_iris(iris_file: Optional[UploadFile], eye_side: str) -> Optional[dict]:
    if not iris_file:
        return None
    try:
        contents = await iris_file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_GRAYSCALE)
        output = iris_pipeline(img_data=image, eye_side=eye_side)
        if output["error"]:
            raise ValueError(f"{eye_side.capitalize()} eye image error")
        return output
    except Exception:
        return None

async def enroll_user(
    left_iris: Optional[UploadFile] = File(None),
    right_iris: Optional[UploadFile] = File(None),
    pcode: str = "",
    cid: str = ""
):
    """
    Enroll user using left and/or right iris images.
    """
    output_L = await process_iris(left_iris, "left")
    output_R = await process_iris(right_iris, "right")

    if not output_L and not output_R:
        raise HTTPException(status_code=400, detail="Neither image contains a valid iris.")
    unique_id = uuid.uuid4().hex
    if output_L and output_R:
        left_path = save_iris_image(left_iris, cid, "L", unique_id)
        print(f"Left iris saved to {left_path}")
        right_path = save_iris_image(right_iris, cid, "R", unique_id)
        print(f"Right iris saved to {right_path}")
        return await EnrollUser_bothiris(output_L=output_L, output_R=output_R, pcode=pcode, cid=cid)
    elif output_L:
        left_path = save_iris_image(left_iris, cid, "L", unique_id)
        print(f"Left iris saved to {left_path}")
        return await EnrollUser_singleiris(output=output_L, eye_side="left", pcode=pcode, cid=cid)
    else:
        right_path = save_iris_image(right_iris, cid, "R", unique_id)
        print(f"Right iris saved to {right_path}")
        return await EnrollUser_singleiris(output=output_R, eye_side="right", pcode=pcode, cid=cid)

async def verify_user(cid: str, left_iris: UploadFile, right_iris: UploadFile):
    """
    Verify iris images against a given citizen ID.
    """
    output_L = await process_iris(left_iris, "left")
    output_R = await process_iris(right_iris, "right")

    if not output_L and not output_R:
        raise HTTPException(status_code=400, detail="Neither image contains a valid iris.")
    if output_L and output_R:
        return await VerifyUser_bothiris(output_L=output_L, output_R=output_R, cid=cid)
    elif output_L:
        return await VerifyUser_singleiris(output=output_L, eye_side="left",cid=cid)
    else:
        return await VerifyUser_singleiris(output=output_R, eye_side="right", cid=cid)

async def search_user(
    left_iris: Optional[UploadFile] = File(None),
    right_iris: Optional[UploadFile] = File(None)
):
    """
    Search for a matching user using iris images.
    """
    output_L = await process_iris(left_iris, "left")
    output_R = await process_iris(right_iris, "right")

    if not output_L and not output_R:
        raise HTTPException(status_code=400, detail="Neither image contains a valid iris.")

    if output_L and output_R:
        return await SearchUser_bothiris(output_L=output_L, output_R=output_R)
    elif output_L:
        return await SearchUser_singleiris(output=output_L, eye_side="left")
    else:
        return await SearchUser_singleiris(output=output_R, eye_side="right")