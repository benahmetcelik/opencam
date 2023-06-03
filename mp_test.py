import cv2
import dlib
import get_model as MyModel

# Yüz tespiti için dlib yüz dedektörünü başlatma
face_detector = dlib.get_frontal_face_detector()

# Webcam'i başlatma
cap = cv2.VideoCapture(0)

while True:
    # Kameradan bir kare okuma
    ret, frame = cap.read()

    if not ret:
        break

    # Kameradan okunan kareyi gri tonlamalı hale getirme
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Yüz tespiti için dlib yüz dedektörünü kullanma
    faces = face_detector(gray)

    for face in faces:

        

        # Yüzün sınırlayıcı kutusunun koordinatlarını al
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

       
        model_result = MyModel.recognize_faces(frame)
        text = model_result[0]
        frame_color = model_result[1]
        cv2.rectangle(frame, (x, y), (x + w, y + h), frame_color, 2)
        cv2.putText(frame, text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, frame_color, 2)

    # Görüntüyü ekranda gösterme
    cv2.imshow("Webcam", frame)

    # 'q' tuşuna basıldığında döngüden çıkma
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakma
cap.release()
cv2.destroyAllWindows()
