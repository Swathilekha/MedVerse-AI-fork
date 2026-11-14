import os
import shutil
import uuid

from fastapi import (
    APIRouter,
    HTTPException,
    UploadFile,
    File,
    Form,
    Depends,
)

from ..config import UPLOAD_DIR
from ..services.ocr_service import extract_text
from ..services.clinical_evidence_service import analyze_report_text
from ..services.ai_service import (
    get_fusion_result,
    get_xray_result,
    get_patient_evidence,
    get_data_status,
)

from ..services.xray_service import predict_xray
from ..services.rag_service import ask_rag

from ..utils.dependencies import get_current_user
from ..models.user import User

router = APIRouter(
    prefix="/ai",
    tags=["AI"]
)


# --------------------------------------------------
# Existing Fusion Endpoint
# --------------------------------------------------

@router.get("/fusion/{patient_id}")
def fusion(patient_id: str, current_user: User = Depends(get_current_user)):

    result = get_fusion_result(
        patient_id
    )

    if result is None:

        raise HTTPException(
            status_code=404,
            detail="Fusion result unavailable"
        )

    return result


# --------------------------------------------------
# Existing X-Ray CSV Lookup
# --------------------------------------------------

@router.get("/xray/{patient_id}")
def xray_result(patient_id: str, current_user: User = Depends(get_current_user)):

    result = get_xray_result(
        patient_id=patient_id
    )

    if result is None:

        raise HTTPException(
            status_code=404,
            detail="X-ray prediction unavailable"
        )

    return result


# --------------------------------------------------
# LIVE X-RAY UPLOAD + PREDICTION
# --------------------------------------------------

@router.post("/xray/upload")
async def upload_xray(
    patient_id: str = Form(...),
    file: UploadFile = File(...), 
    current_user: User = Depends(get_current_user)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No X-ray file provided"
        )

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp"
    }

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported X-ray format. "
                "Use PNG, JPG, JPEG, BMP or WEBP."
            )
        )

    os.makedirs(
        UPLOAD_DIR,
        exist_ok=True
    )

    safe_filename = (
        f"xray_{patient_id}_"
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    save_path = os.path.join(
        UPLOAD_DIR,
        safe_filename
    )

    try:

        with open(
            save_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        result = predict_xray(
            save_path
        )

        return {
            "patient_id": patient_id,

            "file_name":
                safe_filename,

            "file_path":
                save_path,

            "analysis":
                result
        }

    except Exception as e:

        if os.path.exists(save_path):
            os.remove(save_path)

        raise HTTPException(
            status_code=500,
            detail=f"X-ray analysis failed: {str(e)}"
        )


@router.post("/report/upload")
async def upload_report(
    patient_id: str = Form(...),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No medical report provided"
        )

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".webp"
    }

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported report format. "
                "Use PNG, JPG, JPEG, BMP or WEBP."
            )
        )

    os.makedirs(
        UPLOAD_DIR,
        exist_ok=True
    )

    safe_filename = (
        f"report_{patient_id}_"
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    save_path = os.path.join(
        UPLOAD_DIR,
        safe_filename
    )

    try:

        with open(
            save_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # OCR
        ocr_text = extract_text(
            save_path
        )

        if not ocr_text.strip():

            raise HTTPException(
                status_code=422,
                detail="OCR could not extract readable text from the report."
            )

        # Research clinical evidence pipeline
        evidence = analyze_report_text(
            ocr_text
        )

        return {

            "patient_id":
                patient_id,

            "file_name":
                safe_filename,

            "ocr_text":
                ocr_text,

            "evidence":
                evidence

        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Medical report analysis failed: {str(e)}"
        )
# --------------------------------------------------
# Clinical Evidence
# --------------------------------------------------

@router.get("/evidence/{patient_id}")
def evidence(patient_id: str, current_user: User = Depends(get_current_user)):

    result = get_patient_evidence(
        patient_id
    )

    if not result:

        raise HTTPException(
            status_code=404,
            detail="Clinical evidence unavailable"
        )

    return result


# --------------------------------------------------
# AI Service Status
# --------------------------------------------------

@router.get("/status")
def ai_status(current_user: User = Depends(get_current_user)):

    return get_data_status()

from pydantic import BaseModel


class RAGQuestion(BaseModel):
    question: str


@router.post("/rag/ask")
def rag_ask(payload: RAGQuestion, current_user: User = Depends(get_current_user)):

    try:
        return ask_rag(payload.question)

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"RAG service unavailable: {str(e)}"
        )