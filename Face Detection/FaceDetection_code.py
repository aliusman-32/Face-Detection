import cv2 as cv


cam = cv.VideoCapture(0)

#cam.set(3,650) # width 
#cam.set(4,400) # height


while True:
    
    status , frame = cam.read()
    
    
    frame = cv.flip(frame, 1)
    
    
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    
    gray_image = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    
    
    
    face = face_cascade.detectMultiScale(gray_image,1.1,4)
    
    for (x,y,w,h) in face:
        top_left = (x,y)
        bottom_right = (x+w,y+h)
        
        cv.rectangle(frame,top_left,bottom_right,(0,255,0))
            
    cv.imshow("face", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
































