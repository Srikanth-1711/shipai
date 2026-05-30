from fastapi import APIRouter
from typing import Dict, Any
from datetime import datetime
import hashlib
import json
from app.db.feedback_db import load_decision_record, save_record, update_signals, load_all_signals, count_records, overall_rate, get_top_corrections
from app.models.feedback import DecisionRecord

router = APIRouter(prefix="/feedback")

def hash_requirements(req: Dict[str, Any]) -> str:
    # Only hash the fields that should influence this decision
    key_fields = {
        "scale": req.get("scale"),
        "data_location": req.get("data_location"),
        "data_type": req.get("data_type"),
        "privacy_level": req.get("privacy_level")
    }
    return hashlib.md5(
        json.dumps(key_fields, sort_keys=True).encode()
    ).hexdigest()[:8]

@router.post("/accept/{session_id}")
async def record_acceptance(session_id: str):
    """
    Called when user clicks Download without changes.
    Records all decisions as correct.
    """
    record = load_decision_record(session_id)
    if not record:
        return {"status": "error", "message": "Record not found"}
        
    record.outcome = "accepted"
    record.correct_decisions = list(record.config_json.keys())
    record.wrong_decisions = []
    save_record(record)
    
    req_hash = hash_requirements(record.requirements)
    update_signals(record, req_hash)
    
    return {"status": "recorded"}

@router.post("/modify/{session_id}")
async def record_modification(
    session_id: str, 
    modifications: Dict[str, Any]  # e.g. {"vector_db": "pgvector"}
):
    """
    Called when user changes any field in the UI
    before downloading.
    """
    record = load_decision_record(session_id)
    if not record:
        return {"status": "error", "message": "Record not found"}
        
    record.outcome = "modified"
    
    if not hasattr(record, 'modifications') or record.modifications is None:
        record.modifications = {}
        
    for field, new_value in modifications.items():
        record.modifications[field] = {
            "from": record.config_json.get(field),
            "to": new_value
        }
        
    record.wrong_decisions = list(record.modifications.keys())
    record.correct_decisions = [
        k for k in record.config_json.keys()
        if k not in record.modifications
    ]
    
    save_record(record)
    req_hash = hash_requirements(record.requirements)
    update_signals(record, req_hash)
    
    return {"status": "recorded", "signals_updated": True}

@router.post("/abandon/{session_id}")
async def record_abandonment(session_id: str):
    """Called when WebSocket closes without download."""
    record = load_decision_record(session_id)
    if not record:
        return {"status": "error", "message": "Record not found"}
        
    record.outcome = "abandoned"
    save_record(record)
    return {"status": "recorded"}

@router.get("/insights")
async def get_insights() -> dict:
    """
    Returns aggregated intelligence about where
    ShipAI is making wrong decisions.
    """
    signals = load_all_signals()
    
    weak_decisions = [
        {
            "requirements_hash": s.requirements_hash,
            "component": s.component,
            "shipai_choice": s.chosen,
            "acceptance_rate": s.acceptance_rate,
            "users_prefer": s.most_common_correction,
            "sample_size": s.accepted_count + s.rejected_count
        }
        for s in signals
        if s.acceptance_rate < 0.70  # below 70% = needs review
        and (s.accepted_count + s.rejected_count) >= 5
    ]
    
    return {
        "total_sessions": count_records(),
        "overall_acceptance_rate": overall_rate(),
        "weak_decisions": sorted(
            weak_decisions,
            key=lambda x: x["acceptance_rate"]
        ),
        "top_corrections": get_top_corrections()
    }
