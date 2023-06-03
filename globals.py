from pathlib import Path
DEFAULT_ENCODINGS_PATH = Path("face_recognizer/output/encodings.pkl")



def dirsCheck():
    Path("face_recognizer").mkdir(exist_ok=True)
    Path("face_recognizer/training").mkdir(exist_ok=True)
    Path("face_recognizer/output").mkdir(exist_ok=True)
    Path("face_recognizer/validation").mkdir(exist_ok=True)

