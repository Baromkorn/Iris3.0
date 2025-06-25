from fastapi import FastAPI, APIRouter, UploadFile, File, Depends
from typing import Optional
from services_logic.iris_service import verify_user, enroll_user, search_user
from Database.DatabaseCheck.DatabaseCheck import check_available
from routers.API_KEYMAKER import get_api_key
import cv2
import numpy as np

def decode_bytes_image(image_bytes: Optional[bytes]) -> Optional[np.ndarray]:
    if not image_bytes:
        return None
    try:
        image_np = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None
    
image_router = APIRouter(
    dependencies=[Depends(get_api_key)]
                   )

@image_router.post("/enroll_image/")
async def enroll_image(
    iris_L: Optional[UploadFile] = File(None),
    iris_R: Optional[UploadFile] = File(None),
    pcode: str = "",
    cid: str = ""
):
    left = decode_bytes_image(await iris_L.read()) if iris_L else None
    right = decode_bytes_image(await iris_R.read()) if iris_R else None
    res = await enroll_user(left_iris=left, right_iris=right, pcode=pcode, cid=cid)
    status = res.get("status")
    if status == "success" :
        return True
    else :
        return False

@image_router.post("/verify_image/")
async def verify_image(
    iris_L: Optional[UploadFile] = File(None),
    iris_R: Optional[UploadFile] = File(None),
    cid: str = ""
):
    left = decode_bytes_image(await iris_L.read()) if iris_L else None
    right = decode_bytes_image(await iris_R.read()) if iris_R else None
    res = await verify_user(cid, left, right)
    status = res.get("status")
    if status == "success" :
        match = res.get("match")
        if match :
            return True
        else :
            return False
    else :
        return False

@image_router.post("/search_image/")
async def search_image(
    iris_L: Optional[UploadFile] = File(None),
    iris_R: Optional[UploadFile] = File(None)
):
    left = decode_bytes_image(await iris_L.read()) if iris_L else None
    right = decode_bytes_image(await iris_R.read()) if iris_R else None
    res = await search_user(left, right)
    status = res.get("status")
    if status == "success" :
        return [res]
    elif status == "failed" :
        return []


@image_router.get("/check_available/{pcode_or_cid}/")
async def check(pcode_or_cid: str):
    res = await check_available(pcode_or_cid)
    status = res.get("status")
    Used = res.get("available")
    if status == "success" :
        if Used :
            return True
        else :
            return False
    else :
        return False

# Create separate app instance
app = FastAPI(title="Iris Image Upload API")
app.include_router(image_router)
