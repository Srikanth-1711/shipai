import json
from app.db.feedback_db import load_all_signals
from app.services.ollama_service import ollama_service
import chromadb

CHROMA_DIR = "./knowledge_base/chroma_db"

async def optimize_patterns():
    signals = load_all_signals()
    
    # Find weak decisions (<70% acceptance and at least 5 signals)
    weak = [
        s for s in signals 
        if s.acceptance_rate < 0.70 and (s.accepted_count + s.rejected_count) >= 5
    ]
    
    if not weak:
        return "No weak patterns detected requiring optimization."
        
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        col = client.get_collection("shipai_books")
    except Exception:
        return "Warning: ChromaDB collection not found. Cannot retrieve L4 principles."
        
    suggestions = []
    
    for w in weak:
        # Retrieve relevant L4 principles based on the component and top correction
        query_text = f"When to use {w.most_common_correction} for {w.component} vs {w.chosen}"
        results = col.query(
            query_texts=[query_text],
            n_results=3,
            where={"level": "principle"}
        )
        
        principles = []
        if results['documents'] and results['documents'][0]:
            principles = results['documents'][0]
            
        principles_text = "\n\n".join(f"- {p}" for p in principles)
        
        prompt = f"""You are the Lead Architect for ShipAI.
We have detected a flaw in our architectural decision matrix.
For the component '{w.component}', ShipAI consistently chose '{w.chosen}'.
However, users rejected this {(1 - w.acceptance_rate)*100:.1f}% of the time, overwhelmingly preferring '{w.most_common_correction}'.

Context (Requirements Hash): {w.requirements_hash}

Here are some relevant engineering principles from our knowledge base:
{principles_text}

Based on these principles and the user feedback, write a concrete, specific rule that we should add to our decision matrix so that we choose '{w.most_common_correction}' under the right conditions instead of '{w.chosen}'.
Do not be generic. Name the exact tradeoffs.
"""
        
        try:
            # Use ollama_service or direct httpx
            suggestion = await ollama_service.generate(
                prompt=prompt,
                model="qwen2.5:3b",
                temperature=0.3
            )
        except Exception as e:
            suggestion = f"Failed to generate suggestion: {e}"
            
        suggestions.append({
            "component": w.component,
            "chosen": w.chosen,
            "preferred": w.most_common_correction,
            "acceptance_rate": w.acceptance_rate,
            "suggestion": suggestion
        })
        
    return suggestions
