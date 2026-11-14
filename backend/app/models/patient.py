from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from datetime import datetime
from ..database import Base

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    age = Column(Integer, nullable=False)
    gender = Column(String(30), nullable=False)
    medical_history = Column(Text, default="")
    symptoms = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
