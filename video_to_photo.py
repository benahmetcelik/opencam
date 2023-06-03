import cv2
from matplotlib import pyplot as plt

video = cv2.VideoCapture(0)
fps = video.get(cv2.CAP_PROP_FPS)
print(fps)
count = 54
while True:
    ret, frame = video.read()
    
    img = frame
    count += 1
    image_save = cv2.imwrite('images/ahmet_'+str(count)+'.jpg', img)
 
  
    cv2.imshow('Video', img)

    if cv2.waitKey(1) == ord('q'):
        break

# Opening image
