import json
from pathlib import Path
from typing import Dict

DATA_FILE = Path("backend/data_store.json")

class InMemoryStore:
    def __init__(self):
        self.candidates = {}
        self.offers = {}
        self._load()

    def _load(self):
        if DATA_FILE.exists():
            try:
                obj = json.loads(DATA_FILE.read_text())
                self.candidates = obj.get("candidates", {})
                self.offers = obj.get("offers", {})
            except Exception:
                pass

    def _save(self):
        try:
            DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            DATA_FILE.write_text(json.dumps({
                "candidates": self.candidates,
                "offers": self.offers
            }, indent=2))
        except Exception:
            pass

    # candidates
    def add_candidate(self, cand: dict):
        self.candidates[cand["id"]] = cand
        self._save()
    def get_candidate(self, cid: str):
        return self.candidates.get(cid)
    def list_candidates(self):
        return list(self.candidates.values())

    # offers
    def add_offer(self, offer: dict):
        self.offers[offer["id"]] = offer
        self._save()
    def get_offer(self, oid: str):
        return self.offers.get(oid)
    def list_offers(self):
        return list(self.offers.values())

store = InMemoryStore()
