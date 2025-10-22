from fastapi import APIRouter, UploadFile, File, Form
from typing import List
from .models import CandidateCreate, Candidate, OfferCreate, Offer
from .storage import store
from .nlp.extractor import extract_skills_from_text
from .nlp.matcher import combined_score
import pdfplumber
import io

router = APIRouter()

# helpers
def read_pdf_bytes(file_bytes: bytes):
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages = pdf.pages
        for p in pages:
            try:
                text += p.extract_text() or ""
            except Exception:
                pass
    return text

# Candidates
@router.post("/candidates", response_model=Candidate)
def create_candidate(payload: CandidateCreate):
    skills = payload.skills or extract_skills_from_text(payload.text or "")
    cand = Candidate(name=payload.name, text=payload.text or "", skills=skills)
    store.add_candidate(cand.dict())
    return cand

@router.post("/candidates/upload", response_model=Candidate)
async def upload_candidate(name: str = Form(...), file: UploadFile = File(...)):
    data = await file.read()
    text = ""
    if file.filename.lower().endswith(".pdf"):
        text = read_pdf_bytes(data)
    else:
        text = data.decode("utf-8", errors="ignore")
    skills = extract_skills_from_text(text)
    cand = Candidate(name=name, text=text, skills=skills)
    store.add_candidate(cand.dict())
    return cand

@router.get("/candidates")
def list_candidates():
    return store.list_candidates()

@router.get("/candidates/{cid}")
def get_candidate(cid: str):
    c = store.get_candidate(cid)
    if not c:
        return {"error": "not found"}
    return c

# Offers
@router.post("/offers", response_model=Offer)
def create_offer(payload: OfferCreate):
    skills = payload.skills or extract_skills_from_text(payload.description or "")
    off = Offer(title=payload.title, description=payload.description, skills=skills)
    store.add_offer(off.dict())
    return off

@router.get("/offers")
def list_offers():
    return store.list_offers()

@router.get("/offers/{oid}")
def get_offer(oid: str):
    o = store.get_offer(oid)
    if not o:
        return {"error": "not found"}
    return o

# Match: match one candidate to all offers, return top-k
@router.get("/match/{candidate_id}")
def match_candidate(candidate_id: str, top_k: int = 5):
    candidate = store.get_candidate(candidate_id)
    if not candidate:
        return {"error": "candidate not found"}
    offers = store.list_offers()
    results = []
    for o in offers:
        score, breakdown = combined_score(
            candidate_text=candidate.get("text",""),
            candidate_skills=candidate.get("skills", []),
            offer_text=o.get("description",""),
            offer_skills=o.get("skills", [])
        )
        results.append({
            "offer": o,
            "score": score,
            "breakdown": breakdown,
            "common_skills": list(set([s.lower() for s in candidate.get("skills", [])]).intersection(set([s.lower() for s in o.get("skills", [])])))
        })
    results_sorted = sorted(results, key=lambda r: r["score"], reverse=True)
    return {"candidate": candidate, "matches": results_sorted[:top_k]}