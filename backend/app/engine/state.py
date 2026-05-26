from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

class SystemRequirements(TypedDict):
    # Discovery layer - free-form semantic descriptions captured from user conversation
    user_level: str          
    use_case: str            
    data_type: str           
    data_location: str       
    data_change_rate: str    
    scale: str               
    query_type: str          
    privacy_level: str       
    
class AgentState(TypedDict):
    # Conversation
    messages: List[dict]
    user_input: str
    
    # Requirements (built during interview)
    requirements: SystemRequirements
    requirements_complete: bool
    
    # Research (built during Meta-RAG)
    book_insights: List[str]
    github_patterns: List[str]
    research_iterations: int
    research_complete: bool
    
    # Decision (built during Pattern Matcher)
    selected_architecture: str
    config_json: dict
    
    # Output
    explanation: str         # Explainer agent output
    project_path: str        # Where the generated project lives
