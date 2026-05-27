from typing import TypedDict, List, Optional

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

    # Validation (between planner and builder)
    config_validation_errors: List[str]
    
    # Build output (Builder node)
    generated_files: List[str]  # Relative file paths inside project_path
    build_error: str            # Non-empty if codegen failed

    # Verification output (Verifier node)
    verification_report: dict    # VerificationReport.to_dict()
    verification_failed: bool
    verification_metrics: dict

    # Output
    explanation: str         # Explainer agent output
    project_path: str        # Where the generated project lives
