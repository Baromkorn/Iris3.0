from fastapi import APIRouter, UploadFile
from services_logic.iris_service import enroll_user, verify_user, search_user
from Database.DatabaseCreate.DatabaseCreate import CreateDatabase
router = APIRouter()

@router.post("/enrollDatabase")
async def enroll(file: UploadFile, pcode: str, eye_side: str, cid: str):
    res = await enroll_user(file, pcode, eye_side, cid)
    return res

@router.post("/verify")
async def verify(file: UploadFile, pcode: str, eye_side: str):
    return await verify_user(file, pcode, eye_side)

@router.post("/searchDatabase")
async def search(file: UploadFile, eye_side: str, cid: str):
    return await search_user(file, eye_side, cid)

@router.get("/createDatabase")
async def create() :
    return await CreateDatabase()