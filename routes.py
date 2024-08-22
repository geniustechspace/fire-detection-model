import os
from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse
from pathlib import Path

from model.fire import detect_fire, get_best_detection

router = APIRouter()


@router.get("/")
def chat():
    # result = detect_fire(user_input)
    image_path = Path("pic.jpg")
    if not image_path.is_file():
        return {"error": "Image not found on the server"}

    return FileResponse(image_path)


@router.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    # Save the uploaded file to a temporary location
    temp_file = Path(f"temp_{file.filename}")
    with temp_file.open("wb") as buffer:
        buffer.write(await file.read())

    try:
        # Detect fire in the image
        # output_path = detect_fire(temp_file)
        output_path = get_best_detection(temp_file)

        # Return the image with fire detection
        return FileResponse(output_path)
    finally:
        # Delete the temporary file
        if temp_file.exists():
            os.remove(temp_file)
