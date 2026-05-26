import sqlite3
import json
from pathlib import Path
from typing import Optional, List
from app.models.feedback import DecisionRecord, FeedbackSignal

DB_PATH = Path.home() / ".shipai" / "feedback.db"

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decision_records (
            session_id TEXT PRIMARY KEY,
            timestamp TEXT,
            requirements TEXT,
            config_json TEXT,
            outcome TEXT,
            modifications TEXT,
            correct_decisions TEXT,
            wrong_decisions TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            requirements_hash TEXT,
            component TEXT,
            chosen TEXT,
            accepted_count INTEGER DEFAULT 0,
            rejected_count INTEGER DEFAULT 0,
            corrections TEXT,
            UNIQUE(requirements_hash, component, chosen)
        )
    """)
    conn.commit()
    conn.close()

def save_record(record: DecisionRecord):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT OR REPLACE INTO decision_records 
        (session_id, timestamp, requirements, config_json, outcome, modifications, correct_decisions, wrong_decisions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.session_id,
            record.timestamp.isoformat(),
            json.dumps(record.requirements),
            json.dumps(record.config_json),
            record.outcome,
            json.dumps(record.modifications),
            json.dumps(record.correct_decisions),
            json.dumps(record.wrong_decisions)
        )
    )
    conn.commit()
    conn.close()

def load_decision_record(session_id: str) -> Optional[DecisionRecord]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM decision_records WHERE session_id = ?", (session_id,)).fetchone()
    conn.close()
    if not row:
        return None
    from datetime import datetime
    return DecisionRecord(
        session_id=row['session_id'],
        timestamp=datetime.fromisoformat(row['timestamp']),
        requirements=json.loads(row['requirements']),
        config_json=json.loads(row['config_json']),
        outcome=row['outcome'],
        modifications=json.loads(row['modifications']),
        correct_decisions=json.loads(row['correct_decisions']),
        wrong_decisions=json.loads(row['wrong_decisions'])
    )

def update_signals(record: DecisionRecord, requirements_hash: str):
    conn = sqlite3.connect(DB_PATH)
    
    # Process accepted decisions
    for comp in record.correct_decisions:
        chosen = record.config_json.get(comp)
        if isinstance(chosen, dict):
            chosen = chosen.get("choice", str(chosen))
            
        row = conn.execute("SELECT id, accepted_count FROM feedback_signals WHERE requirements_hash=? AND component=? AND chosen=?", (requirements_hash, comp, chosen)).fetchone()
        if row:
            conn.execute("UPDATE feedback_signals SET accepted_count = ? WHERE id = ?", (row[1] + 1, row[0]))
        else:
            conn.execute("INSERT INTO feedback_signals (requirements_hash, component, chosen, accepted_count, rejected_count, corrections) VALUES (?, ?, ?, 1, 0, '{}')", (requirements_hash, comp, chosen))
            
    # Process rejected decisions
    for comp in record.wrong_decisions:
        chosen = record.config_json.get(comp)
        if isinstance(chosen, dict):
            chosen = chosen.get("choice", str(chosen))
        correction_to = record.modifications.get(comp, {}).get("to")
        
        row = conn.execute("SELECT id, rejected_count, corrections FROM feedback_signals WHERE requirements_hash=? AND component=? AND chosen=?", (requirements_hash, comp, chosen)).fetchone()
        if row:
            corrections = json.loads(row[2])
            corrections[correction_to] = corrections.get(correction_to, 0) + 1
            conn.execute("UPDATE feedback_signals SET rejected_count = ?, corrections = ? WHERE id = ?", (row[1] + 1, json.dumps(corrections), row[0]))
        else:
            corrections = {correction_to: 1}
            conn.execute("INSERT INTO feedback_signals (requirements_hash, component, chosen, accepted_count, rejected_count, corrections) VALUES (?, ?, ?, 0, 1, ?)", (requirements_hash, comp, chosen, json.dumps(corrections)))
            
    conn.commit()
    conn.close()

def load_all_signals() -> List[FeedbackSignal]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM feedback_signals").fetchall()
    conn.close()
    
    signals = []
    for row in rows:
        signals.append(FeedbackSignal(
            requirements_hash=row["requirements_hash"],
            component=row["component"],
            chosen=row["chosen"],
            accepted_count=row["accepted_count"],
            rejected_count=row["rejected_count"],
            corrections=json.loads(row["corrections"])
        ))
    return signals

def count_records() -> int:
    conn = sqlite3.connect(DB_PATH)
    count = conn.execute("SELECT COUNT(*) FROM decision_records").fetchone()[0]
    conn.close()
    return count

def overall_rate() -> float:
    conn = sqlite3.connect(DB_PATH)
    accepted = conn.execute("SELECT COUNT(*) FROM decision_records WHERE outcome='accepted'").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM decision_records").fetchone()[0]
    conn.close()
    return accepted / total if total > 0 else 0.0

def get_top_corrections() -> dict:
    signals = load_all_signals()
    top = {}
    for s in signals:
        if s.corrections:
            for k, v in s.corrections.items():
                name = f"{s.component}: {s.chosen} -> {k}"
                top[name] = top.get(name, 0) + v
    return dict(sorted(top.items(), key=lambda item: item[1], reverse=True)[:5])
