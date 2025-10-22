import spacy
import re

# charge spaCy (assure-toi d'avoir téléchargé en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# petite liste de mots-clés tech pour aider (on peut l'agrandir)
COMMON_SKILLS = [
    "python","java","javascript","typescript","angular","react","spring","django",
    "flask","sql","postgresql","mongodb","docker","kubernetes","aws","git",
    "c++","c#","node.js","node","html","css","tensorflow","pytorch","machine learning",
    "nlp","rest","graphql","microservices"
]

def normalize_skill(s: str) -> str:
    return s.strip().lower()

def extract_skills_from_text(text: str):
    text_low = text.lower()
    found = set()

    # 1) match known tokens
    for token in COMMON_SKILLS:
        if re.search(r'\b' + re.escape(token) + r'\b', text_low):
            found.add(token)

    # 2) use spaCy to detect ORG/PRODUCT or noun chunks (heuristic)
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT", "WORK_OF_ART"):
            s = normalize_skill(ent.text)
            if len(s) > 1 and len(s.split()) <= 3:
                found.add(s)

    # 3) heuristique: detect tokens like "Angular", "Spring Boot"
    # we already capture some via COMMON_SKILLS and spaCy

    return sorted(list(found))
