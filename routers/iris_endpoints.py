from fastapi import APIRouter, UploadFile, File, Depends
from pydantic import BaseModel
from typing import Optional
from services_logic.iris_service import verify_user, enroll_user, search_user
from Database.DatabaseCheck.DatabaseCheck import check_available
from routers.API_KEYMAKER import get_api_key
from utils.file_utils import decode_base64_image

router = APIRouter(
    dependencies=[Depends(get_api_key)]
                   )

class IrisPayload_Enroll(BaseModel):
    iris_L: Optional[str] = None
    iris_R: Optional[str] = None
    pcode: str 
    cid: str

class IrisPayload_Verify(BaseModel):
    iris_L: Optional[str] = None
    iris_R: Optional[str] = None
    cid: str

class IrisPayload_search(BaseModel):
    iris_L: Optional[str] = None
    iris_R: Optional[str] = None

@router.post("/")
async def enroll(payload: IrisPayload_Enroll
):
    """
    Enroll a user with optional left and right iris images and identifiers. Duplicates with same 

    Args:
        left_iris (Optional[UploadFile]): File upload for the left iris image(.BMT).
        right_iris (Optional[UploadFile]): File upload for the right iris image(.BMT).
        pcode (str): Personal code or identifier.
        cid (str): Citizen ID.

    Returns:
        dict: Enrollment result containing:
            - status (str): "success" or "failed"
    """
    left = decode_base64_image(payload.iris_L) if payload.iris_L else None
    right = decode_base64_image(payload.iris_R) if payload.iris_R else None
    res = await enroll_user(left_iris=left,
    right_iris=right,
    pcode=payload.pcode,
    cid=payload.cid)
    status = res.get("status")
    if status == "success" :
        return True
    else :
        return False


@router.post("/verify/")
async def verify(
    payload: IrisPayload_Verify
):
    """
    Verify a user's iris to cid in Database, BOTH sides have to match or else it will return false

    Args:
        cid (str): Citizen ID to verify.
        left_iris (UploadFile): Uploaded image(.BMT) for the left iris.
        right_iris (UploadFile): Uploaded image(.BMT) for the right iris.

    Returns:
        dict: Verification result including:
            - status (str): "success" or "failed"
            - message (str): Details about verification
            - match (bool, optional): result of match, if status failed then not returned
            - cid (str, optional): The customer ID if matched
            - reason (str, optional): Reason for failure if any
           
    """
    left = decode_base64_image(payload.iris_L) if payload.iris_L else None
    right = decode_base64_image(payload.iris_R) if payload.iris_R else None
    res = await verify_user(payload.cid ,left, right)
    status = res.get("status")
    if status == "success" :
        match = res.get("match")
        if match :
            return True
        else :
            return False
    else :
        return False

@router.post("/search/")
async def search(
    payload: IrisPayload_search
):
    """
    Search the database for users matching the provided iris images. Using Top-5 results of vector search combined with hamming_distance matcher from open-iris

    Args:
        left_iris (Optional[UploadFile]): File upload for the left iris image.
        right_iris (Optional[UploadFile]): File upload for the right iris image.

    Returns:
        dict: Search results containing:
            - status (str): "success" or "failed"
            - reason (str, optional): Additional info for status failed
            - pcode (str,optional): Personal code of top match
            - cid (str,optional): Citizen ID of top match
            - score (int,optional): normalized score of hamming distance from 1-2000, threshold used is 0.37, so >=740 is a match  
    """
    left = decode_base64_image(payload.iris_L) if payload.iris_L else None
    right = decode_base64_image(payload.iris_R) if payload.iris_R else None
    res = await search_user(left, right)
    status = res.get("status")
    if status == "success" :
        return [res]
    elif status == "failed" :
        return []

    

@router.get("/check_available/{pcode_or_cid}/")
async def check(pcode_or_cid: str):
    """
    Check if a given personal code or Citizen ID is available in the database.

    Args:
        pcode_or_cid (str): The personal code or Citizen ID to check.

    Returns:
        dict: Availability info including:
            - status (str): "success" or "failed"
            - available (bool, optional): True if already in database, False if not found
            - reason (str, optional): Additional details of status failed
    """
    
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