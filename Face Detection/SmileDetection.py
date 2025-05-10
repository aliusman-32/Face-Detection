import cv2 as cv

path = r"Enter your image path here ! "



img = cv.imread(path)  


face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_smile.xml')
    
gray_image = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    
    
    
face = face_cascade.detectMultiScale(gray_image,1.7,15)

for (x,y,w,h) in face:
    top_left = (x,y)
    bottom_right = (x+w,y+h)
    
    cv.rectangle(img,top_left,bottom_right,(0,255,0))
        
cv.imshow("face", img)

cv.waitKey(0)
cv.destroyAllWindows()




   













































