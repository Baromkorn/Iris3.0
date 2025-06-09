from typing import Union,Annotated
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import cv2
import iris
import numpy as np


iris_pipeline = iris.IRISPipeline()
matcher = iris.HammingDistanceMatcher()
'''@asynccontextmanager
async def lifespan(app: FastAPI):
    iris_pipeline = iris.IRISPipeline()
    yield
    iris_pipeline = None'''

app = FastAPI()




@app.post("/enroll")
async def create_upload_file(file: UploadFile, pcode:str,eye_side:str):
    content = await file.read()
    with open(f"{pcode}_{eye_side}.png","wb") as f:
        f.write(content)
    img_pixels = np.frombuffer(content,np.uint8)
    img = cv2.imdecode(img_pixels,cv2.IMREAD_GRAYSCALE)
    output = iris_pipeline(img_data=img, eye_side=eye_side)
    if output["error"] is not None:
        raise HTTPException(status_code=400, detail="Image not contain iris")
    return {"template": len(output["iris_template"].iris_codes), eye_side: eye_side}

@app.post("/verify")
async def verify(file: UploadFile, pcode:str,eye_side:str):
    content = await file.read()
    img_pixels = np.frombuffer(content,np.uint8)
    img = cv2.imdecode(img_pixels,cv2.IMREAD_GRAYSCALE)
    output = iris_pipeline(img_data=img, eye_side=eye_side)
    print(output["error"])
    img_subject = cv2.imread(f"{pcode}_{eye_side}.png", cv2.IMREAD_GRAYSCALE)
    output2 = iris_pipeline(img_data=img_subject, eye_side=eye_side)
    distance = matcher.run(output2["iris_template"],output["iris_template"])
    return {"Verification": str(distance <= 0.37),"distance":distance}
