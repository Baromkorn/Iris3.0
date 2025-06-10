import numpy as np
import cv2
from fastapi import UploadFile, HTTPException
from fast_api.core.iris_setup import iris_pipeline, matcher
from fast_api.utils.file_utils import save_file, load_image

async def enroll_user(file: UploadFile, pcode: str, eye_side: str):
    content = await file.read()
    filename = f"{pcode}_{eye_side}.png"
    save_file(content, filename)
    img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_GRAYSCALE)

    output = iris_pipeline(img_data=img, eye_side=eye_side)
    if output["error"] is not None:
        raise HTTPException(status_code=400, detail="Image does not contain iris")

    return {"template": len(output["iris_template"].iris_codes), "eye_side": eye_side}

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
