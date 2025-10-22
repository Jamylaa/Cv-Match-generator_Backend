from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import uuid4

def new_id():
    return str(uuid4())

class CandidateCreate(BaseModel):
    name: str
    text: Optional[str] = ""
    skills: Optional[List[str]] = []

class Candidate(BaseModel):
    id: str = Field(default_factory=new_id)
    name: str
    text: str = ""
    skills: List[str] = []

class OfferCreate(BaseModel):
    title: str
    description: str
    skills: Optional[List[str]] = []

class Offer(BaseModel):
    id: str = Field(default_factory=new_id)
    title: str
    description: str
    skills: List[str] = []
