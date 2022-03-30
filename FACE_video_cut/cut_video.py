import cv2
import os
def detectFace(img,c):

    img = cv2.imread(img) # 讀取圖檔
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 透過轉換函式轉為灰階影像
    
    # OpenCV 人臉識別分類器
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # 調用偵測識別人臉函式
    faceRects = face_classifier.detectMultiScale(
        grayImg, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    
    # 大於 0 則檢測到人臉
    if len(faceRects):  
        # 框出每一張人臉
        for faceRect in faceRects: 
            x, y, w, h = faceRect
            #cv2.rectangle(img, (x, y), (x + h, y + w), color, 0)
            crop_img = img[y-25:y+w+25, x-25:x+h+25]
            #crop_img = img[y:y+w, x:x+h]
    
    # 將結果圖片輸出
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("cut"+str(c)+'.jpg', crop_img)
#%%每幀(c)張擷取
'''
cap = cv2.VideoCapture('E:\\學校課程\\test0625.mp4')

c=1
timeF = 60

while(True):
    ret, frame = cap.read()
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame', gray)
    
    if c % timeF == 0:
        cv2.imwrite(str(int(c / timeF)) + '.jpg', gray)
    c+=1
    
    if cv2.waitKey(12) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''
#%%辨識擷取

cap = cv2.VideoCapture('E:\\學校課程\\test0625.mp4')

c=0
timeF = 10

while(True):
    ret, frame = cap.read()
    if(c % timeF == 0):
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cv2.imshow('frame', gray)
        cv2.imwrite("r_face"+str(c)+".jpg", gray)
        
        try:
            detectFace("r_face"+str(c)+".jpg",c)
        except:
            pass
        
        fileTest = r""+"r_face"+str(c)+".jpg"
        try:
            os.remove(fileTest)
        except OSError as e:
            print(e)
        else:
            pass
    
    c+=1
    
    if cv2.waitKey(12) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()