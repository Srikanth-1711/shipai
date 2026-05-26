from datetime import datetime, timedelta
from typing import Optional
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.database import get_db
from app.models.user import UserResponse

# Configuration
SECRET_KEY = os.getenv("SHIPAI_JWT_SECRET", "super_secret_jwt_key_change_me_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserResponse:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    async with get_db() as db:
        async with db.execute("SELECT * FROM users WHERE email = ?", (email,)) as cursor:
            user_row = await cursor.fetchone()
            if user_row is None:
                raise credentials_exception
            
            return UserResponse(
                id=user_row["id"],
                email=user_row["email"],
                display_name=user_row["display_name"],
                tier=user_row["tier"],
                created_at=datetime.fromisoformat(user_row["created_at"]) if isinstance(user_row["created_at"], str) else user_row["created_at"]
            )
