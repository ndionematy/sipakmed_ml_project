# api/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import traceback
import os, uuid
from sqlalchemy.orm import Session
from api.database import SessionLocal, Patient, Diagnosis
from api.model import load_keras_model, predict_class
from api.utils  import preprocess_image

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def save_image(contents: bytes, filename: str) -> str:
    """Sauvegarde l’image dans /uploads et renvoie le chemin relatif."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    return file_path

# ---------- Création de l'app ----------
app = FastAPI(
    title="SIPaKMeD – Cell Classification API",
    version="1.0.0",
    description="API pour classer les cellules cervicales à l'aide d'un DenseNet entraîné."
)

# ---------- Charge le modèle DenseNet une seule fois ----------
model = load_keras_model()   # Chemin par défaut : ../models/densenet_ccu.h5

# ---------- Endpoint principal ----------
@app.post("/predict")
async def predict(
    prenom: str = Form(...),
    nom : str = Form(...),
    age       : int  = Form(..., gt=0),
    image     : UploadFile = File(...)
):
    session: Session = SessionLocal()
    """
    Reçoit infos patient + image.
    Renvoie la classe cellulaire prédite.
    """
    try:
        contents = await image.read()
        img_array = preprocess_image(contents)
        cell_type = predict_class(model, img_array)

        # 1️⃣ Enregistrer / retrouver patient
        patient = session.query(Patient).filter_by(
            prenom=prenom, nom=nom, age=age
        ).first()
        if not patient:
            patient = Patient(prenom=prenom,
                            nom=nom,
                            age=age)
            session.add(patient)
            session.commit()   # obtient patient.id

        # 2️⃣ Sauvegarder l’image sur disque
        fname = f"{uuid.uuid4().hex}_{image.filename}"
        img_rel_path = save_image(contents, fname)

        # 3️⃣ Enregistrer le diagnostic
        diag = Diagnosis(patient_id=patient.id,
                        image_path=img_rel_path,
                        cell_type=cell_type)
        session.add(diag)
        session.commit()

        return {
            "patient_id": patient.id,
            "patient": {
                "prenom": patient.prenom,
                "nom": patient.nom,
                "age": patient.age
            },
            "diagnosis": {
                "cell_type": cell_type,
                "image_saved": img_rel_path
            }
        }

    except Exception as e:
        session.rollback()
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        session.close()
