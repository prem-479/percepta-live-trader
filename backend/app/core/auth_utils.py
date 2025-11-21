from passlib.context import CryptContext
from sqlalchemy.orm import Session
from app.db.models import User

pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")


# ----------------------------
# HASH PASSWORD
# ----------------------------
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


# ----------------------------
# VERIFY PASSWORD
# ----------------------------
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ----------------------------
# CREATE USER
# ----------------------------
def create_user(db: Session, username: str, password: str):
    hashed = hash_password(password)
    user = User(username=username, password=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ----------------------------
# AUTHENTICATE USER
# ----------------------------
def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.password):
        return None
    return user
