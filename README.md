# 🏦 AI-Powered KYC Document Verification System

A production-style document verification pipeline that automatically extracts, validates, and cross-checks identity information from multiple documents using AI-powered OCR and Large Language Models.

This project simulates real-world KYC (Know Your Customer) workflows used in banking, fintech, and compliance systems.

---

## ✨ Features

* 🔍 AI-powered OCR using Vision Models
* 🤖 Structured data extraction using LLMs
* 📄 Multi-document verification (ID, Bank Statement, Employment Letter)
* ✅ Strict KYC consistency checks across documents
* ⚙️ Modular, production-style architecture
* 🌐 Web interface for document upload and results
* 📝 Detailed logging and error handling

---

## 🧠 KYC Validation Rules

The system enforces deterministic checks across documents:

* Full name matching (case-insensitive)
* Date of birth consistency
* Address similarity (≥ 70%)
* Phone number matching
* Father’s name verification
* PAN format validation (ABCDE1234F)
* Aadhaar format validation (12 digits)

---

## 🏗️ Architecture

**OCR → Structured Extraction → Cross-Document Verification → Result**

1. Vision model extracts raw text from documents
2. LLM converts unstructured text into structured data
3. Verification engine checks consistency across documents
4. System produces final verification status

---

## 🖥️ Tech Stack

**Backend**

* Python
* Flask
* Google Gemini (Vision + LLM)
* Loguru (logging)
* python-dotenv

**Frontend**

* HTML
* CSS

---

## 📂 Project Structure

```
AI-Document-Verification-System/
│
├── main.py                 # Flask app + pipeline entry point
├── verifier.py             # Verification engine
├── utils.py                # Utilities and logging
│
├── templates/
│   ├── index.html          # Upload page
│   └── result.html         # Result display page
│
├── static/
│   └── style.css           # Styling
│
├── data/                   # Sample input documents
├── output/                 # Generated results
└── .env                    # API keys (not included)
```

---

