from pathlib import Path
import pickle
import face_recognition
from collections import Counter
import PIL.Image
import numpy as np
from PIL import Image
import cv2
import globals



DEFAULT_ENCODINGS_PATH = globals.DEFAULT_ENCODINGS_PATH

globals.dirsCheck()




def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []
    for filepath in Path("face_recognizer/training").glob("*/*"):
        name = filepath.parent.name
        print(f"Encoding: {filepath}")
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)
    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

    #Remove
encode_known_faces()
