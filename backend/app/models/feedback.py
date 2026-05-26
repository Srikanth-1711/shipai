from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class DecisionRecord:
    session_id: str
    timestamp: datetime
    
    # What the system decided
    requirements: dict          # full SystemRequirements
    config_json: dict           # full config ShipAI generated
    
    # What the user actually did
    outcome: str                # accepted | modified | abandoned
    modifications: dict         # fields changed + what they changed to
    
    # Derived signals
    correct_decisions: list[str]
    wrong_decisions: list[str]

@dataclass  
class FeedbackSignal:
    """Aggregated signal for one decision combination"""
    requirements_hash: str      # hash of key requirement fields
    component: str              # e.g. "vector_db"
    chosen: str                 # what ShipAI chose
    accepted_count: int = 0
    rejected_count: int = 0
    corrections: dict = field(default_factory=dict)
    
    @property
    def acceptance_rate(self) -> float:
        total = self.accepted_count + self.rejected_count
        return self.accepted_count / total if total > 0 else 0.0
    
    @property
    def most_common_correction(self) -> Optional[str]:
        if not self.corrections:
            return None
        return max(self.corrections, key=self.corrections.get)
