import asyncio
import sys
import os
import json
import uuid

# Force UTF-8 stdout encoding for Windows compatibility with LLM emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add backend directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from app.engine.orchestrator import app_brain

TEST_CASES = [
    {
        "name": "Cisco router logs",
        "messages": [
            "I want agentic RAG for my team",
            "Cisco router logs, 200 engineers, NFS paths, daily regression RCA"
        ],
        "max_turns": 3,
        "expected": {
            "infra_tier": "enterprise",    # 200 engineers
            "search_type": "hybrid",       # log + semantic queries
            "cache": "redis",              # 200 concurrent users
            "pii": "none",                 # internal logs
        }
    },
    {
        "name": "Restaurant menu",
        "messages": [
            "I want AI for my small restaurant to answer customer questions about our menu"
        ],
        "max_turns": 2,
        "expected": {
            "infra_tier": "minimal",       # small business
            "vector_db": "chroma",         # local, simple
            "cache": "none",               # low traffic
        }
    },
    {
        "name": "HIPAA multi-tenant",
        "messages": [
            "Multi-tenant RAG, 500 enterprise users, HIPAA compliance needed, documents in S3"
        ],
        "max_turns": 2,
        "expected": {
            "pii": "presidio",             # HIPAA = PII required
            "infra_tier": "enterprise",    # 500 users
        }
    },
    {
        "name": "Vague company AI",
        "messages": [
            "I need AI for my company",
            "We have 50 employees, internal HR documents, questions about policies"
        ],
        "max_turns": 3,
        "expected": {
            "template": "rag_chatbot",
            "infra_tier": "standard",      # 50 users = team scale
        }
    },
    {
        "name": "Ambiguous Company AI (needs clarification)",
        "messages": [
            "I need AI for my company"
        ],
        "max_turns": 1,
        "expect_incomplete": True
    }
]

async def run_test_case(case: dict) -> bool:
    print(f"\n==========================================")
    print(f"Running Test Case: {case['name']}")
    print(f"==========================================")
    
    # Clean state for each test case by using a unique thread_id
    thread_id = f"thread_{uuid.uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}
    
    state = {"messages": [], "requirements": {}, "requirements_complete": False}
    turns = 0
    
    for message in case["messages"]:
        print(f"USER: {message}")
        # Append latest user message manually to state
        state["messages"].append({"role": "user", "content": message})
        
        # Invoke LangGraph asynchronously
        state = await app_brain.ainvoke(state, config)
        turns += 1
        
        # Log assistant response if any
        assistant_msgs = [m for m in state.get("messages", []) if m.get("role") == "assistant"]
        if assistant_msgs:
            print(f"AGENT: {assistant_msgs[-1]['content']}")
        print(f"Requirements Confirmed: {state.get('requirements')}")
        print(f"Requirements Complete: {state.get('requirements_complete')}")
        print(f"------------------------------------------")
        
        if state.get("requirements_complete"):
            break
            
    # Check if incomplete is expected
    if case.get("expect_incomplete"):
        assert not state.get("requirements_complete"), \
            f"Expected requirements to be incomplete, but they were complete."
        print(f"[PASS] {case['name']} - Asked clarifying question successfully in {turns} turn(s)")
        return True
        
    assert state.get("requirements_complete"), \
        f"Expected requirements to be complete, but they were incomplete."
    assert turns <= case["max_turns"], \
        f"Took {turns} turns, expected <= {case['max_turns']}"
    
    config_json = state.get("config_json", {})
    print(f"Generated Config: {json.dumps(config_json, indent=2)}")
    
    for field, expected_value in case["expected"].items():
        actual_value = config_json.get(field)
        assert actual_value == expected_value, \
            f"{field}: expected {expected_value}, got {actual_value}"
            
    # For HIPAA case, vector_db must be pgvector or milvus
    if case["name"] == "HIPAA multi-tenant":
        vector_db = config_json.get("vector_db")
        assert vector_db in ("pgvector", "milvus"), \
            f"vector_db: expected pgvector or milvus for enterprise HIPAA, got {vector_db}"
            
    print(f"[PASS] {case['name']} - Passed with correct configuration in {turns} turns")
    return True

async def main():
    print("Starting test suite...")
    all_passed = True
    for case in TEST_CASES:
        try:
            await run_test_case(case)
        except AssertionError as e:
            print(f"[FAIL] Test case failed: {e}")
            all_passed = False
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[FAIL] Test case failed due to exception: {e}")
            all_passed = False
            
    if all_passed:
        print("\n=== ALL TESTS PASSED SUCCESSFULLY! ===")
        sys.exit(0)
    else:
        print("\n=== SOME TESTS FAILED! ===")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
