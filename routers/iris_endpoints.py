from fastapi import APIRouter, UploadFile
from fast_api.services_logic.iris_service import enroll_user, verify_user

router = APIRouter()

@router.post("/enroll")
async def enroll(file: UploadFile, pcode: str, eye_side: str):
    return await enroll_user(file, pcode, eye_side)

@router.post("/verify")
async def verify(file: UploadFile, pcode: str, eye_side: str):
    return await verify_user(file, pcode, eye_side)
