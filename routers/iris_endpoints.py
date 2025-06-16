from fastapi import APIRouter, UploadFile
from services_logic.iris_service import enroll_user, verify_user, search_user

router = APIRouter()

@router.post("/enroll")
async def enroll(file: UploadFile, pcode: str, eye_side: str, cid: str):
    res = await enroll_user(file, pcode, eye_side, cid)
    return res

@router.post("/verify")
async def verify(file: UploadFile, pcode: str, eye_side: str):
    return await verify_user(file, pcode, eye_side)

@router.post("/search")
async def search(file: UploadFile, eye_side: str, cid: str):
    return await search_user(file, eye_side, cid)
