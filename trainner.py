import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()
path='dataSet'

def getImagesWithID(path):
   
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    
    faceSamples=[]
   
    Ids=[]
    
    for imagePath in imagePaths:
        
         pilImage=Image.open(imagePath).convert('L')
         imageNp=np.array(pilImage,'uint8')
        
         Id=int(os.path.split(imagePath)[-1].split(".")[1])
         faceSamples.append(imageNp)
         print (Id)
         Ids.append(Id)
         cv2.imshow("trainning",imageNp)
         cv2.waitKey(10)
    return Ids,faceSamples     
        
    


Ids,faceSamples = getImagesWithID(path)
recognizer.train(faceSamples, np.array(Ids))
recognizer.save('trainner/trainner.yml')
cv2.destroyAllWindows()
