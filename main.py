import os
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv
from loguru import logger
import google.generativeai as genai
from flask import Flask, render_template, request

from utils import setup_logging, time_it
from verifier import DataExtractor, VerificationEngine


# ==============================
# Core Verification System
# ==============================

class DocumentVerificationSystem:

    def __init__(self):
        setup_logging()
        logger.info("Initializing Document Verification System")
        load_dotenv()

        self.vision_client = self._initialize_vision_client()
        self.llm_client, self.llm_provider = self._initialize_llm_client()
        self.data_extractor = DataExtractor(
            self.vision_client,
            self.llm_client,
            self.llm_provider
        )
        self.verification_engine = VerificationEngine()

        logger.info("System initialization complete")

    def _initialize_vision_client(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set in .env file")

        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-pro")

    def _initialize_llm_client(self):
        provider = os.getenv("LLM_PROVIDER", "gemini").lower()

        if provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")

            genai.configure(api_key=api_key)
            client = genai.GenerativeModel("gemini-2.5-pro")
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        return client, provider

    @time_it
    def process_person_documents(self, person_dir: Path) -> Optional[Dict]:

        person_id = person_dir.name
        logger.info(f"Processing documents for {person_id}")

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.pdf'}
        image_files = [f for f in person_dir.iterdir()
                       if f.suffix.lower() in image_extensions]

        if not image_files:
            logger.warning("No document images found")
            return None

        extracted_data = {}

        for idx, image_file in enumerate(image_files, 1):

            doc_key = f"document_{idx}"
            doc_type = self._infer_document_type(image_file.name)

            raw_text = self.data_extractor.extract_text_from_image(
                str(image_file)
            )

            if raw_text:
                structured = self.data_extractor.structure_text_with_llm(
                    raw_text,
                    doc_type
                )
                if structured:
                    extracted_data[doc_key] = structured

        if not extracted_data:
            logger.error("No data extracted")
            return None

        verification_results = self.verification_engine.run_all_verifications(
            extracted_data
        )

        return {
            "person_id": person_id,
            "extracted_data": extracted_data,
            "verification_results": verification_results,
            "overall_status": verification_results["overall_status"]
        }

    def _infer_document_type(self, filename: str) -> str:

        fn = filename.lower()

        if "aadhaar" in fn or "aadhar" in fn:
            return "Government ID"
        elif "pan" in fn:
            return "PAN Card"
        elif "bank" in fn or "statement" in fn:
            return "Bank Statement"
        elif "employment" in fn or "letter" in fn:
            return "Employment Letter"
        return "Unknown Document"


# ==============================
# Flask Web App
# ==============================

app = Flask(__name__)

UPLOAD_ROOT = Path("uploads")
UPLOAD_ROOT.mkdir(exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/verify", methods=["POST"])
def verify():

    # Create unique folder for this user
    person_id = f"P{len(list(UPLOAD_ROOT.iterdir())) + 1}"
    person_dir = UPLOAD_ROOT / person_id
    person_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded files
    for file in request.files.values():
        if file and file.filename:
            file_path = person_dir / file.filename
            file.save(file_path)

    # Run verification system
    system = DocumentVerificationSystem()
    result = system.process_person_documents(person_dir)

    if result:
        status = result["overall_status"]
        data = result["extracted_data"]
    else:
        status = "ERROR"
        data = {}

    # Clean temp files
    shutil.rmtree(person_dir, ignore_errors=True)

    return render_template("result.html",
                           status=status,
                           data=data)


# ==============================
# Run Server
# ==============================

if __name__ == "__main__":
    app.run(debug=True)
