from fastapi import APIRouter, UploadFile, File, Query
from services_logic.iris_service import  verify_user, enroll_user, search_user
from Database.DatabaseCheck.DatabaseCheck import check_available
from typing import Optional, Annotated
router = APIRouter()

@router.post("/")
async def enroll(left_iris: Optional[UploadFile] = File(None),right_iris: Optional[UploadFile]= File(None), pcode: str="", cid: str=""):
    res = await enroll_user(left_iris, right_iris, pcode, cid)
    return res

@router.post("/verify/")
async def verify( cid: str, left_iris: UploadFile,right_iris: UploadFile):
    return await verify_user(cid, left_iris,right_iris)

@router.post("/search/")
async def search(left_iris: Optional[UploadFile] = File(None),right_iris: Optional[UploadFile] = File(None)):
    return await search_user(left_iris, right_iris)


@router.get("/check_available/{pcode_or_cid}/")
async def check(pcode_or_cid : str) : 
    return await check_available(pcode_or_cid)

