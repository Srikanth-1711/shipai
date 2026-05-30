from pydantic import BaseModel, EmailStr
from typing import Optional, Any
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr
    display_name: Optional[str] = None
    tier: str = "free"

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: int
    hashed_password: str
    created_at: datetime

class UserResponse(UserBase):
    id: int
    created_at: datetime

class ProjectBase(BaseModel):
    name: str
    template: str
    config_json: Optional[str] = "{}"
    status: str = "created"

class ProjectCreate(ProjectBase):
    pass

class ProjectResponse(ProjectBase):
    id: int
    user_id: int
    created_at: datetime
