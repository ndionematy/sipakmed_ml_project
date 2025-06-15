from sqlalchemy import (create_engine, Column, Integer, String,
                        DateTime, ForeignKey)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime, os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "ccuBD.db")
engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)

Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    prenom = Column(String(50))
    nom  = Column(String(50))
    age = Column(Integer)
    diagnoses = relationship("Diagnosis", back_populates="patient")

class Diagnosis(Base):
    __tablename__ = "diagnoses"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    image_path = Column(String(255))
    cell_type = Column(String(30))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    patient    = relationship("Patient", back_populates="diagnoses")

# 👉 créer les tables si elles n’existent pas
Base.metadata.create_all(bind=engine)
