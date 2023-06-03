from pathlib import Path
import pickle
import face_recognition
from collections import Counter
import PIL.Image
import numpy as np
import globals


DEFAULT_ENCODINGS_PATH = globals.DEFAULT_ENCODINGS_PATH

globals.dirsCheck()




def recognize_faces(
    image,
    model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    with encodings_location.open(mode="rb") as f:
         loaded_encodings = pickle.load(f)

    im = PIL.Image.fromarray(image)
   
    im = im.convert('RGB')
    input_image = np.array(im)
        
        
    input_face_locations = face_recognition.face_locations(
        input_image, model=model
    )
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )
    
    name = "Unknown"
    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        frame_color = (38,255,10)
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
            frame_color = (0, 0, 255)
       
        return [name,frame_color]
    return [name,frame_color]    

def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0] 
    




