# ==========================================
# TCB: TECHNICAL CONFIGURATION BLOCK
# ==========================================
# Path to your fine-tuned model folder (must contain config.json, tf_model.h5, vocab.txt)
MODEL_PATH = "./best_distilbert_model"

# Path to artifacts (if you used pickle for label encoders or role presets)
# Leave empty string if not used, but structure it for future use.
TAG_MAPPING_PATH = "./tag2idx.pkl"

# Metadata containing tag2idx mapping
METADATA_PATH = "./metadata.json"

# CORS: Add your Vercel frontend URL here after deployment
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://ats-kecil.vercel.app",  # Replace with your actual Vercel URL
    "*",  # Allow all for development
]
# ==========================================

import json
import re
import io
from typing import List, Optional, Tuple
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, TFDistilBertForTokenClassification

# PDF Parsing - Try multiple libraries for best results
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

# Image OCR - for extracting text from images
try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# Supported file extensions
SUPPORTED_PDF_EXTENSIONS = ['.pdf']
SUPPORTED_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
SUPPORTED_EXTENSIONS = SUPPORTED_PDF_EXTENSIONS + SUPPORTED_IMAGE_EXTENSIONS



# ==========================================
# GLOBAL VARIABLES (loaded at startup)
# ==========================================
tokenizer = None
model = None
tag2idx = None
idx2tag = None


# ==========================================
# ROLE PRESETS - Predefined skill sets for common roles
# ==========================================
ROLE_PRESETS = {
    "Backend Engineer": [
        "Python", "Java", "Go", "Node.js", "REST API", "GraphQL",
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Docker", "Kubernetes",
        "AWS", "GCP", "Azure", "Microservices", "CI/CD", "Git"
    ],
    "Frontend Engineer": [
        "JavaScript", "TypeScript", "React", "Vue.js", "Angular",
        "HTML", "CSS", "Sass", "Tailwind CSS", "Webpack", "Vite",
        "Next.js", "Nuxt.js", "Redux", "REST API", "GraphQL", "Git"
    ],
    "Full Stack Developer": [
        "JavaScript", "TypeScript", "Python", "React", "Node.js",
        "Express", "PostgreSQL", "MongoDB", "REST API", "Docker",
        "AWS", "Git", "HTML", "CSS", "Next.js"
    ],
    "Data Scientist": [
        "Python", "R", "SQL", "Machine Learning", "Deep Learning",
        "TensorFlow", "PyTorch", "Pandas", "NumPy", "Scikit-learn",
        "Data Visualization", "Statistics", "NLP", "Computer Vision"
    ],
    "Machine Learning Engineer": [
        "Python", "TensorFlow", "PyTorch", "Keras", "Scikit-learn",
        "MLOps", "Docker", "Kubernetes", "AWS", "GCP", "NLP",
        "Computer Vision", "Deep Learning", "Model Deployment"
    ],
    "DevOps Engineer": [
        "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Terraform",
        "Ansible", "Jenkins", "CI/CD", "Linux", "Shell Scripting",
        "Prometheus", "Grafana", "Git", "Python", "Bash"
    ],
    "Mobile Developer": [
        "Swift", "Kotlin", "Java", "React Native", "Flutter", "Dart",
        "iOS", "Android", "REST API", "Firebase", "Git", "UI/UX"
    ],
    "Data Engineer": [
        "Python", "SQL", "Apache Spark", "Kafka", "Airflow",
        "ETL", "Data Warehousing", "AWS", "GCP", "Hadoop",
        "PostgreSQL", "MongoDB", "Docker", "Kubernetes"
    ],
}


# ==========================================
# PYDANTIC MODELS
# ==========================================
class CandidateResult(BaseModel):
    filename: str
    extracted_skills: List[str]  # All skills extracted
    matched_skills: List[str]    # Skills that match job requirements
    missing_skills: List[str]    # Required skills not found in candidate
    match_score: float           # Percentage of required skills found
    match_count: int             # Number of matched skills
    total_required: int          # Total required skills
    is_top_candidate: bool
    extracted_text_preview: str  # Preview of extracted text for debugging


class MatchResponse(BaseModel):
    role_name: str
    job_skills: List[str]
    total_candidates: int
    top_threshold_percent: float
    candidates: List[CandidateResult]


# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def clean_text(text: str) -> str:
    """
    Clean extracted text by removing excessive whitespace and normalizing.
    This helps the text resemble the JSON format the model was trained on.
    """
    # Replace multiple newlines with single space
    text = re.sub(r'\n+', ' ', text)
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text


def extract_text_with_pdfplumber(pdf_bytes: bytes) -> str:
    """Extract text using pdfplumber - better for structured PDFs."""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            text_parts = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return " ".join(text_parts)
    except Exception as e:
        print(f"pdfplumber failed: {e}")
        return ""


def extract_text_with_pymupdf(pdf_bytes: bytes) -> str:
    """Extract text using PyMuPDF (fitz)."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Use "text" mode for better extraction
            text_parts.append(page.get_text("text"))
        doc.close()
        return " ".join(text_parts)
    except Exception as e:
        print(f"PyMuPDF failed: {e}")
        return ""


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF bytes using multiple methods for best results.
    Tries pdfplumber first (better for structured text), then PyMuPDF as fallback.
    """
    text = ""
    
    # Try pdfplumber first (usually better for CVs)
    if HAS_PDFPLUMBER:
        text = extract_text_with_pdfplumber(pdf_bytes)
        if text and len(text.strip()) > 50:
            print("✅ Used pdfplumber for extraction")
            return clean_text(text)
    
    # Fallback to PyMuPDF
    if HAS_PYMUPDF:
        text = extract_text_with_pymupdf(pdf_bytes)
        if text and len(text.strip()) > 50:
            print("✅ Used PyMuPDF for extraction")
            return clean_text(text)
    
    # If both fail or return too little text
    if not text or len(text.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Failed to extract text from PDF. Please ensure the PDF contains readable text (not scanned images)."
        )
    
    return clean_text(text)


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Extract text from image bytes using OCR (pytesseract).
    Supports: JPG, JPEG, PNG, BMP, TIFF, WEBP
    """
    if not HAS_OCR:
        raise HTTPException(
            status_code=400,
            detail="OCR is not available. Please install pytesseract and Pillow."
        )
    
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed (for PNG with alpha channel, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Run OCR
        text = pytesseract.image_to_string(image, lang='eng')
        
        if not text or len(text.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from image. Please ensure the image contains readable text."
            )
        
        print(f"✅ Used OCR for extraction, got {len(text)} chars")
        return clean_text(text)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"OCR failed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from a file based on its extension.
    Supports PDFs and images (JPG, PNG, etc.)
    """
    ext = '.' + filename.lower().split('.')[-1] if '.' in filename else ''
    
    if ext in SUPPORTED_PDF_EXTENSIONS:
        return extract_text_from_pdf(file_bytes)
    elif ext in SUPPORTED_IMAGE_EXTENSIONS:
        return extract_text_from_image(file_bytes)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )


def extract_skills_from_text(text: str) -> List[str]:
    """
    Run NER inference on text to extract SKILL entities.
    Uses BIO tagging scheme: B-SKILL (beginning), I-SKILL (inside), O (outside)
    
    For long texts, we process in chunks to handle max_length limitation.
    """
    global tokenizer, model, idx2tag

    
    if tokenizer is None or model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    all_skills = []
    
    # Split text into chunks if too long (rough estimate: 200 words per chunk)
    words = text.split()
    chunk_size = 200  # words per chunk
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    for chunk in chunks:
        if not chunk.strip():
            continue
            
        # Tokenize the input text
        encoding = tokenizer(
            chunk,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="tf",
            return_offsets_mapping=True
        )
        
        # Get offset mapping for token-to-text alignment
        offset_mapping = encoding.pop("offset_mapping").numpy()[0]
        
        # Run inference
        outputs = model(**encoding)
        predictions = outputs.logits.numpy()
        predicted_ids = np.argmax(predictions, axis=-1)[0]
        
        # Extract skills using BIO tagging
        current_skill = []
        
        for idx, pred_id in enumerate(predicted_ids):
            # Skip special tokens (offset is (0, 0))
            if offset_mapping[idx][0] == 0 and offset_mapping[idx][1] == 0:
                if current_skill:
                    skill_text = "".join(current_skill).strip()
                    if skill_text and len(skill_text) > 1:
                        all_skills.append(skill_text)
                    current_skill = []
                continue
            
            tag = idx2tag.get(pred_id, "O")
            start, end = offset_mapping[idx]
            token_text = chunk[start:end]
            
            if tag == "B-SKILL":
                # Start of a new skill
                if current_skill:
                    skill_text = "".join(current_skill).strip()
                    if skill_text and len(skill_text) > 1:
                        all_skills.append(skill_text)
                current_skill = [token_text]
            elif tag == "I-SKILL" and current_skill:
                # Continuation of current skill
                current_skill.append(token_text)
            else:
                # Outside tag - save current skill if exists
                if current_skill:
                    skill_text = "".join(current_skill).strip()
                    if skill_text and len(skill_text) > 1:
                        all_skills.append(skill_text)
                    current_skill = []
        
        # Don't forget the last skill
        if current_skill:
            skill_text = "".join(current_skill).strip()
            if skill_text and len(skill_text) > 1:
                all_skills.append(skill_text)
    
    # Clean up skills - remove duplicates and normalize
    cleaned_skills = []
    seen = set()
    for skill in all_skills:
        # Normalize: strip
        normalized = skill.strip()
        if normalized.lower() not in seen and len(normalized) > 1:
            seen.add(normalized.lower())
            cleaned_skills.append(normalized)
    
    return cleaned_skills


def normalize_skill(skill: str) -> str:
    """Normalize a skill string for comparison."""
    # Lowercase and strip
    s = skill.lower().strip()
    # Remove common punctuation
    s = re.sub(r'[.,;:\-_/\\()[\]{}]', ' ', s)
    # Remove extra spaces
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def calculate_match_score(job_skills: List[str], candidate_skills: List[str]) -> Tuple[float, List[str], List[str]]:
    """
    Calculate match score based on how many required skills the candidate has.
    
    Logic:
    - If job requires [Python, JavaScript, React] (3 skills)
    - And candidate has [Python, JavaScript, React, Docker, AWS]
    - Score = 3/3 = 100% (all required skills found)
    
    Returns: (score, matched_skills, missing_skills)
    """
    if not job_skills:
        return 0.0, [], []
    
    if not candidate_skills:
        return 0.0, [], job_skills
    
    # Normalize all skills for comparison
    job_skills_normalized = {normalize_skill(s): s for s in job_skills}
    candidate_skills_normalized = {normalize_skill(s): s for s in candidate_skills}
    
    matched = []
    missing = []
    
    for norm_skill, original_skill in job_skills_normalized.items():
        found = False
        
        # Check for exact match
        if norm_skill in candidate_skills_normalized:
            found = True
        else:
            # Check for partial match (skill is contained in candidate skill or vice versa)
            for cand_norm, cand_original in candidate_skills_normalized.items():
                # Check if job skill is part of candidate skill or vice versa
                if norm_skill in cand_norm or cand_norm in norm_skill:
                    found = True
                    break
                # Also check individual words match
                job_words = set(norm_skill.split())
                cand_words = set(cand_norm.split())
                if job_words and cand_words and (job_words & cand_words):
                    # At least one significant word matches
                    if len(job_words & cand_words) >= 1 and any(len(w) > 2 for w in job_words & cand_words):
                        found = True
                        break
        
        if found:
            matched.append(original_skill)
        else:
            missing.append(original_skill)
    
    # Calculate score as percentage of required skills found
    score = (len(matched) / len(job_skills)) * 100
    
    return round(score, 2), matched, missing


# ==========================================
# FASTAPI APP
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and tokenizer at startup."""
    global tokenizer, model, tag2idx, idx2tag
    
    print("🚀 Loading DistilBERT NER model...")
    print(f"   PDF Libraries: pdfplumber={HAS_PDFPLUMBER}, PyMuPDF={HAS_PYMUPDF}")
    print(f"   OCR (Image): pytesseract={HAS_OCR}")
    
    try:
        # Load tokenizer
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        print("✅ Tokenizer loaded")
        
        # Load model
        model = TFDistilBertForTokenClassification.from_pretrained(MODEL_PATH)
        print("✅ Model loaded")
        
        # Load tag mapping from metadata
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
            tag2idx = metadata.get("tag2idx", {"O": 0, "B-SKILL": 1, "I-SKILL": 2})
        
        idx2tag = {v: k for k, v in tag2idx.items()}
        print(f"✅ Tag mapping loaded: {tag2idx}")
        
        print("🎉 Model ready for inference!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise
    
    yield
    
    print("👋 Shutting down...")


app = FastAPI(
    title="Intelligent ATS API",
    description="API for CV parsing, skill extraction, and job matching using DistilBERT NER",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Intelligent ATS API is running",
        "version": "2.1.0",
        "capabilities": {
            "pdf": HAS_PDFPLUMBER or HAS_PYMUPDF,
            "ocr": HAS_OCR,
            "pdfplumber": HAS_PDFPLUMBER,
            "pymupdf": HAS_PYMUPDF,
        },
        "supported_formats": SUPPORTED_EXTENSIONS,
        "available_roles": list(ROLE_PRESETS.keys())
    }


@app.get("/roles")
async def get_roles():
    """Get available role presets and their skills."""
    return {
        "roles": ROLE_PRESETS
    }


@app.post("/match-cvs", response_model=MatchResponse)
async def match_cvs(
    files: List[UploadFile] = File(..., description="PDF files of candidate CVs"),
    role_name: str = Form(..., description="Role name (e.g., 'Backend Engineer')"),
    additional_skills: str = Form(default="", description="Comma-separated additional skills")
):
    """
    Main endpoint for CV matching.
    
    Scoring Logic (FIXED):
    - Score = (matched skills / required skills) * 100
    - If job needs [A, B, C] and candidate has [A, B, C, D, E] → Score = 100%
    - If job needs [A, B, C] and candidate has [A, B, D, E] → Score = 66.67%
    """
    
    # Validate files
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Get role preset skills
    role_skills = ROLE_PRESETS.get(role_name, [])
    
    # Parse additional skills (comma-separated or JSON)
    extra_skills = []
    if additional_skills:
        try:
            # Try parsing as JSON array first
            extra_skills = json.loads(additional_skills)
            if not isinstance(extra_skills, list):
                extra_skills = [additional_skills]
        except json.JSONDecodeError:
            # Fall back to comma-separated
            extra_skills = [s.strip() for s in additional_skills.split(",") if s.strip()]
    
    # Combine job skills (remove duplicates)
    job_skills = list(dict.fromkeys(role_skills + extra_skills))
    
    if not job_skills:
        raise HTTPException(
            status_code=400,
            detail=f"No skills found for role '{role_name}'. Please provide additional_skills or use a valid role name."
        )
    
    # Process each CV
    candidates = []
    
    for file in files:
        # Get file extension
        filename = file.filename.lower()
        ext = '.' + filename.split('.')[-1] if '.' in filename else ''
        
        # Validate file type
        if ext not in SUPPORTED_EXTENSIONS:
            continue  # Skip unsupported files
        
        try:
            # Read file bytes
            file_bytes = await file.read()
            
            # Extract text from PDF or Image
            text = extract_text_from_file(file_bytes, file.filename)
            text_preview = text[:300] + "..." if len(text) > 300 else text
            
            if not text or len(text) < 20:
                candidates.append(CandidateResult(
                    filename=file.filename,
                    extracted_skills=[],
                    matched_skills=[],
                    missing_skills=job_skills,
                    match_score=0.0,
                    match_count=0,
                    total_required=len(job_skills),
                    is_top_candidate=False,
                    extracted_text_preview="[No text extracted]"
                ))
                continue
            
            # Extract skills using NER
            extracted_skills = extract_skills_from_text(text)
            
            # Calculate match score with new logic
            match_score, matched_skills, missing_skills = calculate_match_score(job_skills, extracted_skills)
            
            candidates.append(CandidateResult(
                filename=file.filename,
                extracted_skills=extracted_skills,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                match_score=match_score,
                match_count=len(matched_skills),
                total_required=len(job_skills),
                is_top_candidate=False,  # Will be set after sorting
                extracted_text_preview=text_preview
            ))
            
        except Exception as e:
            # Log error but continue processing other files
            print(f"Error processing {file.filename}: {e}")
            candidates.append(CandidateResult(
                filename=file.filename,
                extracted_skills=[],
                matched_skills=[],
                missing_skills=job_skills,
                match_score=0.0,
                match_count=0,
                total_required=len(job_skills),
                is_top_candidate=False,
                extracted_text_preview=f"[Error: {str(e)}]"
            ))
    
    # Sort by match score (descending)
    candidates.sort(key=lambda x: x.match_score, reverse=True)
    
    # Mark top 10% as top candidates
    if candidates:
        top_threshold = max(1, len(candidates) // 10)  # At least 1 candidate
        for i, candidate in enumerate(candidates):
            candidate.is_top_candidate = i < top_threshold
    
    return MatchResponse(
        role_name=role_name,
        job_skills=job_skills,
        total_candidates=len(candidates),
        top_threshold_percent=10.0,
        candidates=candidates
    )


@app.post("/extract-skills")
async def extract_skills_endpoint(
    file: UploadFile = File(..., description="PDF or image file to extract skills from")
):
    """
    Extract skills from a single file (PDF or image) without matching.
    Useful for testing and debugging the NER model.
    Returns full extracted text and all skills.
    """
    filename = file.filename.lower()
    ext = '.' + filename.split('.')[-1] if '.' in filename else ''
    
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    # Read and extract text
    file_bytes = await file.read()
    text = extract_text_from_file(file_bytes, file.filename)
    
    # Extract skills
    skills = extract_skills_from_text(text)
    
    return {
        "filename": file.filename,
        "file_type": "image" if ext in SUPPORTED_IMAGE_EXTENSIONS else "pdf",
        "extracted_text_full": text,  # Full text for debugging
        "extracted_text_preview": text[:500] + "..." if len(text) > 500 else text,
        "extracted_skills": skills,
        "skill_count": len(skills)
    }


# For running with: python app.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)  # Hugging Face uses port 7860
