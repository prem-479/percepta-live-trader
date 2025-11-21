from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from fastapi.security import OAuth2PasswordRequestForm

from app.db.database import get_db
from app.core.auth_utils import create_user, authenticate_user
from app.core.security import create_access_token
from app.db.models import User

router = APIRouter()

@router.post("/signup")
def signup(username: str, password: str, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    create_user(db, username, password)
    return {"message": "User created successfully!"}


@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(),
          db: Session = Depends(get_db)):

    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    access_token = create_access_token({"sub": user.username})

    return {"access_token": access_token, "token_type": "bearer"}
