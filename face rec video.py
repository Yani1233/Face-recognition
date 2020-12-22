import os
import face_recognition
import matplotlib.pyplot as plt
import cv2
import random
 

 
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface.xml')
dirFace = 'cropped_face'

# Create if there is no cropped face directory
if not os.path.exists(dirFace):
    os.mkdir(dirFace)
    print("Directory " , dirFace ,  " Created ")
else:    
    print("Directory " , dirFace ,  " has found.")


KNOWN_FACES_DIR = "known_faces" 
#UNKNOWN_FACES_DIR = "unknown_faces"
video = cv2.VideoCapture(0)

known_faces = []
known_names = []

print("Processing Known Faces")

dictionary = {}

for name in os.listdir(KNOWN_FACES_DIR):
  dictionary[name] = False
  for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
    image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
    encoding = face_recognition.face_encodings(image)[0]
    known_faces.append(encoding)
    known_names.append(name)

print("Processing Unknown Faces")

while True:
  
  #print(filename)
  #image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
  
  ret , image = video.read()
  locations = face_recognition.face_locations(image , model = 'cnn') 
  encodings = face_recognition.face_encodings(image , locations)
  #image = cv2.cvtColor(image , cv2.COLOR_RGB2BGR)

  for face_encoding,face_location in zip(encodings,locations):
    results = face_recognition.compare_faces(known_faces , face_encoding , 0.5)
    match = None
    if True in results:
      match = known_names[results.index(True)]
      print(f"Match found : {match}")

      #top_left = (face_location[3], face_location[0])
      #bottom_right = (face_location[2], face_location[1])
      #a = cv2.rectangle(image, top_left, bottom_right, (0,255,0), 2)
      #sub_face = a
      
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.1, 4)
      for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_color = image[y:y + h, x:x + w] 
      
      if dictionary[match] == False: 
        randomnum = random.randint(0,10000)
        FaceFileName = "cropped_face/" + str(match) + "_"+ str(randomnum) +".jpg" # folder path and random name image
        cv2.imwrite(FaceFileName,roi_color)
        print('Face saved')
        dictionary[match] = True

  cv2.imshow('Match', image)
  if cv2.waitKey(1) & 0xFF == ord("q"):
      break
 
  #cv2.destroyAllWindows()

for filename in os.listdir(dirFace):
    print(filename)    
    image = cv2.imread(f"{dirFace}/{filename}")
    match = filename.split('_')[0]
    path = os.path.join(KNOWN_FACES_DIR,match)
    cv2.imwrite(f"{path}/{filename}",image)