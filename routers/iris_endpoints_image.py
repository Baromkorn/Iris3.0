from fastapi import FastAPI, APIRouter, UploadFile, File, Depends
from typing import Optional
from services_logic.iris_service import verify_user, enroll_user, search_user
from Database.DatabaseCheck.DatabaseCheck import check_available
from routers.API_KEYMAKER import get_api_key
import cv2
import numpy as np
import os
from multiprocessing.connection import Client # Tao

# ******************************************************************************
# Load model
#print('Load a model')
#model = joblib.load('/home/suradej/tao/mlp_prediction.pkl')

def read_image(image_path):
    """Read an image from the specified path."""
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return gray_image

def send_task(input_template , address=('localhost', 6000), authkey=b'sabig'):
    conn = Client(address, authkey=authkey)
    #print("[Client] Sending array...")
    conn.send(input_template)
    result = conn.recv()
    conn.close()
    return result

# Shift the bits of the template
def shiftbits_int(template, noshifts):
    templatenew = np.zeros(template.shape, dtype=np.uint8)
    width = template.shape[1]
    s = 2 * np.abs(noshifts)
    p = width - s

    # If no shift is needed, return the original template
    if noshifts == 0:
        templatenew = template

    # If the shift is positive, shift the bits to the right
    elif noshifts < 0:
        x = np.arange(p)
        templatenew[:, x] = template[:, s + x]
        x = np.arange(p, width)
        templatenew[:, x] = template[:, x - p]

    # If the shift is negative, shift the bits to the left
    else:
        x = np.arange(s, width)
        templatenew[:, x] = template[:, x - s]
        x = np.arange(s)
        templatenew[:, x] = template[:, p + x]

    return templatenew
#******************************************************************************

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

@image_router.post("/enroll/")
async def enroll_image(
    iris_L: Optional[UploadFile] = File(None),
    iris_R: Optional[UploadFile] = File(None),
    cid: str = ""
):
    left = decode_bytes_image(await iris_L.read()) if iris_L else None
    right = decode_bytes_image(await iris_R.read()) if iris_R else None
    #res = await enroll_user(left_iris=left, right_iris=right, pcode=pcode, cid=cid)
    res = await enroll_user(left_iris=left, right_iris=right, pcode="", cid=cid)
    status = res.get("status")
    if status == "success" :
        return True
    else :
        return False

@image_router.post("/verify/")
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

@image_router.post("/search/")
async def search_image(
    iris_L: Optional[UploadFile] = File(None),
    iris_R: Optional[UploadFile] = File(None)
):
    left = decode_bytes_image(await iris_L.read()) if iris_L else None
    right = decode_bytes_image(await iris_R.read()) if iris_R else None
    res = await search_user(left, right)
    status = res.get("status")
    matches = res.get("results")
    if status == "success" :
        return matches
    elif status == "failed" :
        return []

@image_router.post("/search2/")
async def search_template(
    template: Optional[UploadFile] = File(None)
):
    image_bytes = await template.read() if template else None
    if not image_bytes:
        return False
    try:
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_GRAYSCALE)
        #print(image.shape)
        #return True
    except Exception:
        return False
    #left = decode_bytes_image(await iris_L.read()) if iris_L else None
    #right = decode_bytes_image(await iris_R.read()) if iris_R else None
    #res = await search_user(left, right)
    #status = res.get("status")
    #matches = res.get("results")
    #init_path = '/home/suradej'
    packed_shifted_template1 = []
    #count = 0
    #f2 = 0
    #i2 = 0
    #image = read_image(os.path.join(init_path, 'tao', \
    #    f'CASIA-IrisV2/device2/00{str(f2).zfill(2)}_template/00{str(f2).zfill(2)}_0{str(i2).zfill(2)}.bmp'))[:, :, 0]

    packed_shifted_template = []
    for l, shifts in enumerate(range(-8, 9)):
        shifted_template = shiftbits_int(image, shifts)
        packed_shifted_template.append(np.packbits(shifted_template.flatten()))

    packed_shifted_template1 = np.asarray(packed_shifted_template)
    result = send_task(packed_shifted_template1)
    print(result)

    return result

@image_router.get("/check_available/{pcode_or_cid}")
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
