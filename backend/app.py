MODEL_PATH = "./best_distilbert_model"
TAG_MAPPING_PATH = "./tag2idx.pkl"
METADATA_PATH = "./metadata.json"
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://ats-kecil.vercel.app",
    "*"
]

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

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

SUPPORTED_EXTENSIONS = ['.pdf']

tokenizer = None
model = None
tag2idx = None
idx2tag = None

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


class CandidateResult(BaseModel):
    filename: str
    extracted_skills: List[str]
    matched_skills: List[str]
    missing_skills: List[str]
    match_score: float
    match_count: int
    total_required: int
    is_top_candidate: bool
    extracted_text_preview: str


class MatchResponse(BaseModel):
    role_name: str
    job_skills: List[str]
    total_candidates: int
    top_threshold_percent: float
    candidates: List[CandidateResult]


def clean_text(text: str) -> str:
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def extract_text_with_pdfplumber(pdf_bytes: bytes) -> str:
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
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_parts.append(page.get_text("text"))
        doc.close()
        return " ".join(text_parts)
    except Exception as e:
        print(f"PyMuPDF failed: {e}")
        return ""


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    text = ""
    
    if HAS_PDFPLUMBER:
        text = extract_text_with_pdfplumber(pdf_bytes)
        if text and len(text.strip()) > 50:
            return clean_text(text)
    
    if HAS_PYMUPDF:
        text = extract_text_with_pymupdf(pdf_bytes)
        if text and len(text.strip()) > 50:
            return clean_text(text)
    
    if not text or len(text.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Failed to extract text from PDF."
        )
    
    return clean_text(text)


def extract_skills_from_text(text: str) -> List[str]:
    global tokenizer, model, idx2tag
    
    if tokenizer is None or model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    all_skills = []
    words = text.split()
    chunk_size = 200
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    for chunk in chunks:
        if not chunk.strip():
            continue
            
        encoding = tokenizer(
            chunk,
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="tf",
            return_offsets_mapping=True
        )
        
        offset_mapping = encoding.pop("offset_mapping").numpy()[0]
        outputs = model(**encoding)
        predictions = outputs.logits.numpy()
        predicted_ids = np.argmax(predictions, axis=-1)[0]
        
        current_skill = []
        
        for idx, pred_id in enumerate(predicted_ids):
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
                if current_skill:
                    skill_text = "".join(current_skill).strip()
                    if skill_text and len(skill_text) > 1:
                        all_skills.append(skill_text)
                current_skill = [token_text]
            elif tag == "I-SKILL" and current_skill:
                current_skill.append(token_text)
            else:
                if current_skill:
                    skill_text = "".join(current_skill).strip()
                    if skill_text and len(skill_text) > 1:
                        all_skills.append(skill_text)
                    current_skill = []
        
        if current_skill:
            skill_text = "".join(current_skill).strip()
            if skill_text and len(skill_text) > 1:
                all_skills.append(skill_text)
    
    cleaned_skills = []
    seen = set()
    for skill in all_skills:
        normalized = skill.strip()
        if normalized.lower() not in seen and len(normalized) > 1:
            seen.add(normalized.lower())
            cleaned_skills.append(normalized)
    
    return cleaned_skills


def normalize_skill(skill: str) -> str:
    s = skill.lower().strip()
    s = re.sub(r'[.,;:\-_/\\()\[\]{}]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def calculate_match_score(job_skills: List[str], candidate_skills: List[str]) -> Tuple[float, List[str], List[str]]:
    if not job_skills:
        return 0.0, [], []
    
    if not candidate_skills:
        return 0.0, [], job_skills
    
    job_skills_normalized = {normalize_skill(s): s for s in job_skills}
    candidate_skills_normalized = {normalize_skill(s): s for s in candidate_skills}
    
    matched = []
    missing = []
    
    for norm_skill, original_skill in job_skills_normalized.items():
        found = False
        
        if norm_skill in candidate_skills_normalized:
            found = True
        else:
            for cand_norm, cand_original in candidate_skills_normalized.items():
                if norm_skill in cand_norm or cand_norm in norm_skill:
                    found = True
                    break
                job_words = set(norm_skill.split())
                cand_words = set(cand_norm.split())
                if job_words and cand_words and (job_words & cand_words):
                    if len(job_words & cand_words) >= 1 and any(len(w) > 2 for w in job_words & cand_words):
                        found = True
                        break
        
        if found:
            matched.append(original_skill)
        else:
            missing.append(original_skill)
    
    score = (len(matched) / len(job_skills)) * 100
    return round(score, 2), matched, missing


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, tag2idx, idx2tag
    
    print("Loading DistilBERT NER model...")
    print(f"PDF Libraries: pdfplumber={HAS_PDFPLUMBER}, PyMuPDF={HAS_PYMUPDF}")
    
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        print("Tokenizer loaded")
        
        model = TFDistilBertForTokenClassification.from_pretrained(MODEL_PATH)
        print("Model loaded")
        
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
            tag2idx = metadata.get("tag2idx", {"O": 0, "B-SKILL": 1, "I-SKILL": 2})
        
        idx2tag = {v: k for k, v in tag2idx.items()}
        print(f"Tag mapping loaded: {tag2idx}")
        print("Model ready for inference!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    yield
    print("Shutting down...")


app = FastAPI(
    title="Intelligent ATS API",
    description="API for CV parsing and skill extraction using DistilBERT NER",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": "healthy",
        "message": "Intelligent ATS API is running",
        "version": "1.0.0",
        "pdf_support": HAS_PDFPLUMBER or HAS_PYMUPDF,
        "supported_formats": SUPPORTED_EXTENSIONS,
        "available_roles": list(ROLE_PRESETS.keys())
    }


@app.get("/roles")
async def get_roles():
    return {"roles": ROLE_PRESETS}


@app.post("/match-cvs", response_model=MatchResponse)
async def match_cvs(
    files: List[UploadFile] = File(..., description="PDF files of candidate CVs"),
    role_name: str = Form(..., description="Role name"),
    additional_skills: str = Form(default="", description="Comma-separated additional skills")
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    role_skills = ROLE_PRESETS.get(role_name, [])
    
    extra_skills = []
    if additional_skills:
        try:
            extra_skills = json.loads(additional_skills)
            if not isinstance(extra_skills, list):
                extra_skills = [additional_skills]
        except json.JSONDecodeError:
            extra_skills = [s.strip() for s in additional_skills.split(",") if s.strip()]
    
    job_skills = list(dict.fromkeys(role_skills + extra_skills))
    
    if not job_skills:
        raise HTTPException(
            status_code=400,
            detail=f"No skills found for role '{role_name}'."
        )
    
    candidates = []
    
    for file in files:
        filename = file.filename.lower()
        ext = '.' + filename.split('.')[-1] if '.' in filename else ''
        
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        
        try:
            file_bytes = await file.read()
            text = extract_text_from_pdf(file_bytes)
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
            
            extracted_skills = extract_skills_from_text(text)
            match_score, matched_skills, missing_skills = calculate_match_score(job_skills, extracted_skills)
            
            candidates.append(CandidateResult(
                filename=file.filename,
                extracted_skills=extracted_skills,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                match_score=match_score,
                match_count=len(matched_skills),
                total_required=len(job_skills),
                is_top_candidate=False,
                extracted_text_preview=text_preview
            ))
            
        except Exception as e:
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
    
    candidates.sort(key=lambda x: x.match_score, reverse=True)
    
    if candidates:
        top_threshold = max(1, len(candidates) // 10)
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
    file: UploadFile = File(..., description="PDF file to extract skills from")
):
    filename = file.filename.lower()
    ext = '.' + filename.split('.')[-1] if '.' in filename else ''
    
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    file_bytes = await file.read()
    text = extract_text_from_pdf(file_bytes)
    skills = extract_skills_from_text(text)
    
    return {
        "filename": file.filename,
        "extracted_text_preview": text[:500] + "..." if len(text) > 500 else text,
        "extracted_skills": skills,
        "skill_count": len(skills)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
