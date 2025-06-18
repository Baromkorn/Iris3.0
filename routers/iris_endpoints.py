from fastapi import APIRouter, UploadFile
from services_logic.iris_service import enroll_user_singleiris, verify_user, search_user_singleiris, enroll_user_bothiris, search_user_bothiris
from Database.DatabaseCreate.DatabaseCreate import CreateDatabase
router = APIRouter()

@router.post("/enroll/single-iris")
async def enroll(file: UploadFile, pcode: str, eye_side: str, cid: str):
    res = await enroll_user_singleiris(file, pcode, eye_side, cid)
    return res

@router.post("/enroll/both-iris")
async def enroll(left_iris: UploadFile,right_iris: UploadFile, pcode: str, cid: str):
    res = await enroll_user_bothiris(left_iris, right_iris, pcode, cid)
    return res

@router.post("/verify")
async def verify(file: UploadFile, pcode: str, eye_side: str):
    return await verify_user(file, pcode, eye_side)

@router.post("/search/single-iris")
async def search(file: UploadFile, eye_side: str):
    return await search_user_singleiris(file, eye_side)

@router.post("/search/both-iris")
async def search(left_iris: UploadFile,right_iris: UploadFile):
    return await search_user_bothiris(left_iris, right_iris)

@router.get("/createDatabase")
async def create() :
    return await CreateDatabase()