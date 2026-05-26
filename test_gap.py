import asyncio
import sys
import os
import json
from langchain_community.chat_models import ChatOllama

sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))
from app.config import settings

async def main():
    llm = ChatOllama(model=settings.OLLAMA_DEFAULT_MODEL, temperature=0.1)
    
    extract_prompt = """You are extracting requirements from a conversation.
Your job is to capture what the user ACTUALLY SAID in their own words. Do not translate, normalize, or invent values.
For each field, extract the specific user-provided details. If a field is not mentioned, set it to "".

Return a JSON object with these fields:
- "use_case": the main goal or task of the AI
- "data_type": the type of data or content mentioned (e.g., logs, documents, menu, etc.). If the user mentions any specific data, documents, log files, menus, or databases, extract that content type into "data_type".
- "data_location": where the data is stored
- "data_change_rate": how often the data changes
- "scale": the scale of the team, system, or users
- "query_type": the nature of user queries
- "privacy_level": compliance or privacy requirements
- "user_level": technical expertise of the target users

Capture only concrete details from the conversation below. If the user did not specify a detail, keep it as "". Do not use placeholder example values.
"""

    cases = {
        "Restaurant Menu": "user: I want AI for my small restaurant to answer customer questions about our menu",
    }

    for name, conv in cases.items():
        resp = await llm.ainvoke(extract_prompt + "\n\nConversation:\n" + conv, format="json")
        print(f"\n--- {name} ---")
        print(resp.content)

if __name__ == "__main__":
    asyncio.run(main())
