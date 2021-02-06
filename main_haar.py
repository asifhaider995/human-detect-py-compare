import os
import cv2
import numpy as np

def main():
    body_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
    cap = cv2.VideoCapture("data/3.mp4")
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (600, 400))

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        body = body_cascade.detectMultiScale2(frame, 1.1, 5, flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in body[0]:
            conf = body[1][0][0]
            count += 1
            if conf > 9:
                name = "frame"+str(count)+".jpg"
                cv2.imwrite("", frame)
                text = f"{conf*10:.2f}%"
                cv2.putText(frame, text, (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow("Detecting Humans", frame)
        print(count)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()