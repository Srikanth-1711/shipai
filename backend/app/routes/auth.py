from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
import sqlite3

from app.core.database import get_db
from app.core.security import (
    get_password_hash,
    verify_password,
    create_access_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from app.models.user import UserCreate, UserResponse

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse)
async def register(user: UserCreate):
    """Register a new user."""
    hashed_password = get_password_hash(user.password)
    
    async with get_db() as db:
        try:
            await db.execute(
                "INSERT INTO users (email, hashed_password, display_name) VALUES (?, ?, ?)",
                (user.email, hashed_password, user.display_name)
            )
            await db.commit()
            
            # Fetch the created user
            async with db.execute("SELECT * FROM users WHERE email = ?", (user.email,)) as cursor:
                row = await cursor.fetchone()
                return UserResponse(
                    id=row["id"],
                    email=row["email"],
                    display_name=row["display_name"],
                    tier=row["tier"],
                    created_at=row["created_at"]
                )
        except sqlite3.IntegrityError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 compatible token login, get an access token for future requests."""
    async with get_db() as db:
        async with db.execute("SELECT * FROM users WHERE email = ?", (form_data.username,)) as cursor:
            user = await cursor.fetchone()
            
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: UserResponse = Depends(get_current_user)):
    """Get the currently logged in user details."""
    return current_user
