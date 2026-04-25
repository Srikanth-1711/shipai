"""
ShipAI — AI Architecture Advisor
The brain that thinks like a senior AI engineer.
Analyzes user's problem → recommends the right AI architecture.
Knows all RAG types, agent patterns, fine-tuning approaches, and when to use each.
"""
import logging
from typing import Optional
from app.services.ollama_service import ollama_service

logger = logging.getLogger(__name__)

# Complete knowledge base of AI patterns and when to use them
AI_PATTERNS = {
    # ─── RAG Types ───
    "naive_rag": {
        "name": "Naive RAG",
        "category": "rag",
        "description": "Simple retrieve-then-generate. Best for basic Q&A over documents.",
        "when_to_use": [
            "Simple FAQ or documentation Q&A",
            "Small to medium document collections (<1000 docs)",
            "Straightforward factual questions",
            "Low latency requirements",
        ],
        "components": ["vector_db", "embedding_model", "llm", "document_loader"],
        "complexity": "low",
        "min_hardware_tier": "minimal",
    },
    "advanced_rag": {
        "name": "Advanced RAG",
        "category": "rag",
        "description": "Pre-retrieval optimization + post-retrieval reranking for better accuracy.",
        "when_to_use": [
            "Complex queries requiring nuanced answers",
            "Large document collections (1000+ docs)",
            "Need higher accuracy than naive RAG",
            "Documents with varied structures",
        ],
        "components": ["vector_db", "embedding_model", "llm", "reranker", "query_optimizer", "chunking_strategies"],
        "complexity": "medium",
        "min_hardware_tier": "basic",
    },
    "agentic_rag": {
        "name": "Agentic RAG",
        "category": "rag",
        "description": "RAG with an AI agent that decides WHEN and HOW to retrieve. Self-correcting.",
        "when_to_use": [
            "Complex multi-step questions",
            "Need to decide between multiple data sources",
            "Self-correcting retrieval (retry if answer is poor)",
            "Dynamic query reformulation needed",
        ],
        "components": ["vector_db", "embedding_model", "llm", "agent_framework", "tool_use", "self_reflection"],
        "complexity": "high",
        "min_hardware_tier": "standard",
    },
    "graph_rag": {
        "name": "Graph RAG",
        "category": "rag",
        "description": "Combines knowledge graphs with vector search for relationship-aware retrieval.",
        "when_to_use": [
            "Data with complex relationships (org charts, supply chains)",
            "Need to traverse connections between entities",
            "Multi-hop reasoning over linked data",
            "Domain-specific knowledge with hierarchies",
        ],
        "components": ["vector_db", "graph_db", "embedding_model", "llm", "entity_extractor"],
        "complexity": "high",
        "min_hardware_tier": "standard",
    },
    "self_rag": {
        "name": "Self-RAG",
        "category": "rag",
        "description": "Model decides if retrieval is needed, then self-evaluates answer quality.",
        "when_to_use": [
            "Mix of questions (some need retrieval, some don't)",
            "Need to reduce unnecessary retrievals",
            "Quality-critical applications",
            "Want model to assess its own confidence",
        ],
        "components": ["vector_db", "embedding_model", "llm", "self_evaluation", "retrieval_classifier"],
        "complexity": "high",
        "min_hardware_tier": "standard",
    },
    "corrective_rag": {
        "name": "Corrective RAG (CRAG)",
        "category": "rag",
        "description": "Evaluates retrieved docs for relevance, falls back to web search if needed.",
        "when_to_use": [
            "Documents may not cover all topics",
            "Need fallback to external knowledge",
            "Quality-critical with relevance scoring",
            "Enterprise applications needing high reliability",
        ],
        "components": ["vector_db", "embedding_model", "llm", "relevance_scorer", "web_search_fallback"],
        "complexity": "high",
        "min_hardware_tier": "standard",
    },
    "multimodal_rag": {
        "name": "Multi-modal RAG",
        "category": "rag",
        "description": "RAG across text, images, tables, and other modalities.",
        "when_to_use": [
            "Documents contain images, charts, or tables",
            "Need to answer questions about visual content",
            "Medical imaging, blueprints, product catalogs",
            "Mixed content types in knowledge base",
        ],
        "components": ["vector_db", "multimodal_embeddings", "vision_model", "llm", "document_parser"],
        "complexity": "very_high",
        "min_hardware_tier": "high",
    },
    "hybrid_rag": {
        "name": "Hybrid RAG",
        "category": "rag",
        "description": "Combines keyword search (BM25) with semantic search for best of both worlds.",
        "when_to_use": [
            "Mix of keyword-heavy and semantic queries",
            "Technical documentation with specific terms",
            "Legal or compliance documents",
            "When pure semantic search misses exact matches",
        ],
        "components": ["vector_db", "bm25_index", "embedding_model", "llm", "fusion_ranker"],
        "complexity": "medium",
        "min_hardware_tier": "basic",
    },

    # ─── Agent Types ───
    "single_agent": {
        "name": "Single AI Agent",
        "category": "agent",
        "description": "One agent with tools that can take actions autonomously.",
        "when_to_use": [
            "Simple automation tasks",
            "Single-domain tool usage (search, calculator, API calls)",
            "Chatbot with action capabilities",
            "Personal assistant features",
        ],
        "components": ["llm", "tool_definitions", "agent_framework", "memory"],
        "complexity": "medium",
        "min_hardware_tier": "basic",
    },
    "multi_agent": {
        "name": "Multi-Agent System",
        "category": "agent",
        "description": "Multiple specialized agents collaborating on complex tasks.",
        "when_to_use": [
            "Complex workflows with different roles",
            "Need for debate/consensus between perspectives",
            "Task decomposition across specialties",
            "Simulating team collaboration",
        ],
        "components": ["llm", "agent_framework", "orchestrator", "inter_agent_communication", "shared_memory"],
        "complexity": "high",
        "min_hardware_tier": "standard",
    },
    "react_agent": {
        "name": "ReAct Agent",
        "category": "agent",
        "description": "Reason + Act pattern — agent thinks step-by-step before each action.",
        "when_to_use": [
            "Tasks requiring planning before action",
            "Multi-step problem solving",
            "Need transparent reasoning chain",
            "Research and analysis tasks",
        ],
        "components": ["llm", "tool_definitions", "reasoning_engine", "action_executor"],
        "complexity": "medium",
        "min_hardware_tier": "basic",
    },
    "autonomous_agent": {
        "name": "Autonomous Workflow Agent",
        "category": "agent",
        "description": "Fully autonomous agent that plans, executes, and iterates without human input.",
        "when_to_use": [
            "Background processing pipelines",
            "Data collection and reporting",
            "Monitoring and alerting systems",
            "Automated content generation pipelines",
        ],
        "components": ["llm", "planner", "executor", "evaluator", "long_term_memory", "error_recovery"],
        "complexity": "very_high",
        "min_hardware_tier": "high",
    },

    # ─── Fine-tuning ───
    "lora_finetuning": {
        "name": "LoRA Fine-tuning",
        "category": "fine_tuning",
        "description": "Parameter-efficient fine-tuning — trains small adapters, not the full model.",
        "when_to_use": [
            "Domain-specific language or terminology",
            "Specific output format requirements",
            "Limited compute resources",
            "Want to maintain base model capabilities",
        ],
        "components": ["base_model", "training_data", "lora_adapters", "training_pipeline"],
        "complexity": "high",
        "min_hardware_tier": "standard",
    },
    "qlora_finetuning": {
        "name": "QLoRA Fine-tuning",
        "category": "fine_tuning",
        "description": "Quantized LoRA — fine-tune on consumer GPUs with 4-bit quantization.",
        "when_to_use": [
            "Limited GPU memory (4-8GB VRAM)",
            "Same use cases as LoRA but with less hardware",
            "Cost-effective fine-tuning",
        ],
        "components": ["base_model_quantized", "training_data", "qlora_adapters", "training_pipeline"],
        "complexity": "high",
        "min_hardware_tier": "basic",
    },
    "full_finetuning": {
        "name": "Full Fine-tuning",
        "category": "fine_tuning",
        "description": "Train all model parameters. Maximum customization, requires significant compute.",
        "when_to_use": [
            "Need maximum model customization",
            "Large training dataset available",
            "Have access to multiple GPUs",
            "Building a domain-specific model from scratch",
        ],
        "components": ["base_model", "large_training_data", "multi_gpu_setup", "training_pipeline"],
        "complexity": "very_high",
        "min_hardware_tier": "ultra",
    },

    # ─── MCP & Tools ───
    "mcp_server": {
        "name": "MCP Server (Tool Integration)",
        "category": "tools",
        "description": "Model Context Protocol server — gives AI agents access to external tools and APIs.",
        "when_to_use": [
            "AI needs to interact with external services",
            "Database queries, API calls, file operations",
            "Need standardized tool interface",
            "Building extensible agent systems",
        ],
        "components": ["mcp_server", "tool_definitions", "api_connectors", "auth_handler"],
        "complexity": "medium",
        "min_hardware_tier": "basic",
    },
}

# System prompt for the AI Advisor
ADVISOR_SYSTEM_PROMPT = """You are ShipAI's AI Architecture Advisor — a senior AI engineer with deep expertise in:
- All types of RAG (Naive, Advanced, Agentic, Graph, Self-RAG, CRAG, Multi-modal, Hybrid)
- AI Agents (Single, Multi-agent, ReAct, Autonomous)
- Fine-tuning (LoRA, QLoRA, Full)
- MCP servers and tool integration
- Production infrastructure (rate limiting, caching, load balancing, circuit breakers, message queues)

Your job: Analyze the user's problem and recommend the RIGHT AI architecture.

Rules:
1. Ask clarifying questions if the problem is vague
2. Always explain WHY you recommend a specific architecture
3. Consider the user's hardware limitations
4. Recommend the simplest solution that solves the problem
5. If multiple approaches work, rank them with trade-offs
6. Be specific — name exact patterns, not generic advice

Output format:
- PRIMARY RECOMMENDATION: The best architecture for this use case
- WHY: Clear reasoning
- COMPONENTS NEEDED: List of technical components
- ALTERNATIVES: Other viable approaches with trade-offs
- HARDWARE REQUIREMENTS: Minimum specs needed
"""


async def analyze_use_case(
    description: str,
    hardware_tier: str = "basic",
    model: str = None,
) -> dict:
    """
    Analyze a user's use case and recommend the optimal AI architecture.
    Uses local LLM + knowledge base for recommendations.
    """
    # Build context from knowledge base
    patterns_context = "\n".join(
        f"- {p['name']} ({p['category']}): {p['description']} | Use when: {', '.join(p['when_to_use'][:2])} | Complexity: {p['complexity']} | Min hardware: {p['min_hardware_tier']}"
        for p in AI_PATTERNS.values()
    )

    prompt = f"""AVAILABLE AI PATTERNS:
{patterns_context}

USER'S HARDWARE TIER: {hardware_tier}
(Only recommend patterns compatible with this tier or lower)

USER'S USE CASE:
{description}

Analyze this use case and provide your recommendation. Be specific and actionable."""

    result = await ollama_service.generate(
        prompt=prompt,
        system=ADVISOR_SYSTEM_PROMPT,
        model=model,
        temperature=0.3,  # Lower temp for more focused recommendations
        max_tokens=1500,
    )

    if "error" in result:
        return {"error": result["error"]}

    # Also do rule-based matching for quick suggestions
    keyword_matches = _keyword_match(description)

    return {
        "analysis": result.get("response", ""),
        "model_used": result.get("model", "unknown"),
        "inference_time_ms": result.get("total_duration_ms", 0),
        "quick_matches": keyword_matches,
        "hardware_tier": hardware_tier,
    }


def _keyword_match(description: str) -> list[dict]:
    """Quick rule-based pattern matching for instant suggestions."""
    desc_lower = description.lower()
    matches = []

    keyword_map = {
        "naive_rag": ["faq", "documentation", "simple q&a", "basic chatbot", "search documents"],
        "advanced_rag": ["large documents", "complex queries", "reranking", "accuracy"],
        "agentic_rag": ["multi-step", "self-correcting", "dynamic", "multiple sources"],
        "graph_rag": ["relationships", "knowledge graph", "connected data", "hierarchy", "org chart"],
        "hybrid_rag": ["keyword search", "technical docs", "legal", "compliance", "exact match"],
        "self_rag": ["confidence", "self-evaluate", "quality critical"],
        "corrective_rag": ["web search fallback", "incomplete data", "reliability"],
        "multimodal_rag": ["images", "charts", "tables", "visual", "photos", "diagrams"],
        "single_agent": ["automation", "simple tool", "api calls", "personal assistant"],
        "multi_agent": ["team", "collaboration", "multiple roles", "workflow", "departments"],
        "react_agent": ["reasoning", "step-by-step", "planning", "research"],
        "autonomous_agent": ["background", "monitoring", "automated pipeline", "no human input"],
        "lora_finetuning": ["domain-specific", "custom language", "fine-tune", "specialized"],
        "qlora_finetuning": ["limited gpu", "cheap fine-tuning", "4-bit", "consumer gpu"],
        "mcp_server": ["external api", "database", "tool integration", "mcp"],
    }

    for pattern_key, keywords in keyword_map.items():
        score = sum(1 for kw in keywords if kw in desc_lower)
        if score > 0:
            pattern = AI_PATTERNS[pattern_key]
            matches.append({
                "pattern": pattern["name"],
                "category": pattern["category"],
                "relevance_score": score,
                "complexity": pattern["complexity"],
                "min_hardware": pattern["min_hardware_tier"],
                "description": pattern["description"],
            })

    matches.sort(key=lambda x: x["relevance_score"], reverse=True)
    return matches[:5]  # Top 5 matches


def get_all_patterns() -> dict:
    """Return the complete knowledge base of AI patterns."""
    categorized = {}
    for key, pattern in AI_PATTERNS.items():
        cat = pattern["category"]
        if cat not in categorized:
            categorized[cat] = []
        categorized[cat].append({
            "id": key,
            "name": pattern["name"],
            "description": pattern["description"],
            "complexity": pattern["complexity"],
            "min_hardware_tier": pattern["min_hardware_tier"],
            "when_to_use": pattern["when_to_use"],
        })
    return categorized
