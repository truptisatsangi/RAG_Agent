import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from pypdf import PdfReader
from io import BytesIO
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI()
logger = logging.getLogger(__name__)

async def upload_pdf(file: UploadFile):
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=415, detail="File type not supported")
    
    # Read uploaded PDF
    pdf_bytes = await file.read()
    pdf_reader = PdfReader(BytesIO(pdf_bytes))

    # Extract Text
    texts = []
    for page in pdf_reader.pages:
        texts.append(page.extract_text() or "")

    if len(texts) < 1:
        raise HTTPException(400, "No text found in PDF")
    
    return {
        "filename": file.filename,
        "pages": len(pdf_reader.pages),
        "text": "\n".join(texts)
    }


# Load and Chunk the data
@app.post("/chunks")
async def make_chunks(file: UploadFile = File(...)):
    try:
        result  = await upload_pdf(file)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap= 100)
        all_chunks = text_splitter.split_text(result["text"])
    
    except HTTPException:
        raise
    except Exception:
        logger.exception("Chunk creation failed")
        raise HTTPException(500, "Internal Server Error")
    
    return {
        "num_chunks": len(all_chunks),
        "chunks": all_chunks
    }




