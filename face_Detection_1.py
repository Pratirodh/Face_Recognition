
import cv2
from mtcnn.mtcnn import MTCNN
from predict import predict
from text_to_speech import text_to_speech
detector = MTCNN()
import time

video = cv2.VideoCapture(0)

if (video.isOpened() == False):
    print("Web Camera not detected")
while (True):
    ret, frame = video.read()
    cv2.imwrite("imageframe.jpg",frame)
    
    frame = cv2.flip(frame, 1)
    # print(type(frame))
    if ret == True:
        location = detector.detect_faces(frame)
        if len(location) > 0:
            for face in location:
                font = cv2.FONT_HERSHEY_SIMPLEX
                x, y, width, height = face['box']
                x2, y2 = x + width, y + height
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
                t1 = time.time()
                predicted_name =  predict('./imageframe.jpg')
                cv2.putText(frame, 
                    predicted_name, 
                    (x, y), 
                    font, 0.5, 
                    (0, 128, 0), 
                    2, 
                    cv2.LINE_4
                )
                text_to_speech(predicted_name)
                t2 = time.time()
                print(t2 - t1)
        cv2.imshow("Output",frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()