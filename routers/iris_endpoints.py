from fastapi import APIRouter, UploadFile, File
from services_logic.iris_service import  verify_user, enroll_user, search_user
from Database.DatabaseCheck.DatabaseCheck import CheckDatabase
from typing import Optional
router = APIRouter()

@router.post("/")
async def enroll(left_iris: Optional[UploadFile] = File(None),right_iris: Optional[UploadFile]= File(None), pcode: str="", cid: str=""):
    res = await enroll_user(left_iris, right_iris, pcode, cid)
    return res

@router.post("/verify/")
async def verify( cid: str, left_iris: Optional[UploadFile] = File(None),right_iris: Optional[UploadFile]= File(None)):
    return await verify_user(cid, left_iris,right_iris)

@router.post("/search/")
async def search(left_iris: Optional[UploadFile] = File(None),right_iris: Optional[UploadFile] = File(None)):
    return await search_user(left_iris, right_iris)

@router.get("/check_available/")
async def check() :
    return await CheckDatabase()