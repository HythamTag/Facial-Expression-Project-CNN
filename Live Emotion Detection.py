#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 
import numpy as np
from keras.models import load_model
import time
from mtcnn.mtcnn import MTCNN
path = "weights/"

def Detect_Faces(image, model):
    return face_detection_model.detect_faces(image)

def Draw_Bounding_BOX(Face_points, image, color_box):
    overlay_face = image.copy()
    overlay_top = image.copy()
    alpha_face=0.8
    alpha_tope=0.5
    shift =1
    x_point, y_point, width, height = Face_points
    
    # print(Face_points)
    cv2.rectangle(overlay_face,
              (Face_points[0], Face_points[1]),
              (Face_points[0]+Face_points[2], Face_points[1] + Face_points[3]),
              color_box,
              2)
        

    #cv2.rectangle(overlay_top, (x_point - shift // 2, y_point - height//20), (x_point + width + shift // 2, y_point - height//5), (255,128,0), -1)
    cv2.rectangle(overlay_top, (x_point - shift, y_point - shift ), (x_point + width + shift, y_point - height//5 - shift), (255,128,0), -1)
    
    cv2.addWeighted(overlay_face, alpha_face, image, 1 - alpha_face, 0, image)
    cv2.addWeighted(overlay_top, alpha_tope, image, 1 - alpha_tope, 0, image)
    
def put_emoji(image,Face_points,Emoji_images,emotion_num):

    x_point, y_point, width, height = Face_points
    shift = 1
    height_rec = 0.2*height
    offset = int(height_rec*0.05)
    
    emotion_size = int(height_rec - offset*2)
    
    x2 = int(x_point - offset + width + shift)
    x1 = int(x2 - emotion_size )
    
    y1 = int(y_point - offset - shift)
    y2 = int(y1 - emotion_size)

    emoji_img = cv2.resize(Emoji_images[emotion_num],(int(emotion_size),int(emotion_size)))
    
    mask =np.array((emoji_img==255), np.uint8)

    image[y2:y1,x1:x2] =  image[y2:y1,x1:x2]*mask + emoji_img*(1-mask)  # dont forget in lost [smaller : larger] no the reverse
    return image
    
    
    
def ROI_Face_Frame(image,Face_points):
    x_point, y_point, width, height = Face_points
    x_offset=int(width/20.)
    y_offset=int(height/20.)
    x1=x_point + x_offset
    x2=x_point + width - x_offset
    y1=y_point - y_offset
    y2=y_point + height 
    face = image[y1:y2,x1:x2]
    return face

#load models 
model1 = path+('weights.51-1.19.hdf5')
model2 =path+('weights.37-0.91.hdf5')
model3 =path+('model_weights.h5')
model4 =path+('weights.08-1.19.hdf5')
model5 = path+('mini_XCEPTION_KDEF.hdf5')
model6 = path+('tiny_XCEPTION_KDEF.hdf5')
model7 = path+('simple_CNN.985-0.66.hdf5')
model8 = path+('simple_CNN.530-0.65.hdf5')
model9 = path+('fer2013_mini_XCEPTION.102-0.66.hdf5')
model10 = path+('weights.51-0.313-conv-32-nodes-0-dense-1568549063.hdf5')

model11 = path+('weights.37-0.29-4-conv-256-nodes-0-dense-1568581634.hdf5')
model12 = path+('weights.46-0.28-4-conv-128-nodes-1-dense-1568590843.hdf5')
model13 = path+('weights.13-0.29-5-conv-64-nodes-0-dense-1568685072-Trial2.hdf5')
model_sep = path+('weights.73-0.83last.hdf5')
weight_last = path+('weights.41-0.24-3-conv-64-nodes-2-dense-1569112505.hdf5')
vvlast = path+('weights.30-0.23-4-conv-128-nodes-2-dense-1569285696.hdf5')
emotion_classifier = load_model(vvlast, compile=False)



# In[ ]:



face_detection_model = MTCNN()

cv2.namedWindow('Live')
video_capture = cv2.VideoCapture(0)
#EMOTIONS_LIST = ['angry', 'happy', 'sad', 'surprise', 'neutral']
EMOTIONS_LIST = ['angry', 'happy', 'sad', 'surprise', 'neutral']
Emoji_images = [cv2.imread('Emojis\{}.png'.format(no)) for no in range(1,6)]
size = (int(video_capture.read()[1].shape[1]*1.8),int(video_capture.read()[1].shape[0]*1.8))

# thanks = cv2.resize(cv2.imread('thanks.png') , size )
# cv2.imshow('Live',thanks)
# cv2.waitKey(3000)


while True:
    bgr_frame = video_capture.read()[1]

    gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

    face_points = Detect_Faces(bgr_frame,face_detection_model)
    
    
    if face_points != []:
        for person in face_points:
            box = person['box']
            if box[2]*box[3] > 10000:
               # print(one_face_points[2]*one_face_points[3] )
                Draw_Bounding_BOX(box,bgr_frame,[0,0,0])
                
                ROI_Face=ROI_Face_Frame(gray_frame,box)
                #cv2.imshow('face', ROI_Face)
                ROI_Face_Predict = cv2.resize(ROI_Face.copy(),(48,48))/255.
                
                ROI_Face_Predict=ROI_Face_Predict[np.newaxis, :, :, np.newaxis]
                
               # print(ROI_Face_Predict.shape)
                custom = (emotion_classifier.predict(ROI_Face_Predict)*100.).tolist()[0]
                
#                 print(custom)
                custom.pop(1)
                custom.pop(1)
#                 print(custom)
                emotion_num = np.argmax(custom)
                
                put_emoji(bgr_frame, box, Emoji_images, emotion_num)
                
                emotion = EMOTIONS_LIST[emotion_num] + " " + str(int(np.max(custom))) + " %"
                
                cv2.putText(bgr_frame, emotion, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                
       # print(custom)
    
    bgr_frame=cv2.resize(bgr_frame,(int(bgr_frame.shape[1]*1.8),int(bgr_frame.shape[0]*1.8)))
    cv2.imshow('Live', bgr_frame)
    #cv2.imshow('grey', gray_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




