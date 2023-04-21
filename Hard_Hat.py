#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install ultralyticsplus')
get_ipython().system('pip install opencv-python')


# In[2]:


from ultralyticsplus import YOLO, render_result
import cv2
from PIL import Image

from cv2 import imshow
from cv2 import imwrite


# In[3]:


# set image
#image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'

    
def PPE(image):
    # load model
    #model = YOLO('keremberke/yolov8m-protective-equipment-detection')
    model = YOLO('keremberke/yolov8m-hard-hat-detection')
    # set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image



    # perform inference
    results = model.predict(image)

    # observe results
    print(results[0].boxes)
    render = render_result(model=model, image=image, result=results[0])
    render.show()


# In[4]:


def take_photo(image='photo.jpg', quality=0.8):

    # program to capture single image from webcam in python
  
    # importing OpenCV library
    from cv2 import imshow
    from cv2 import imwrite
  
    # initialize the camera
    # If you have multiple camera connected with 
    # current device, assign a value in cam_port 
    # variable according to that
    cam_port = 0
    cam = cv2.VideoCapture(cam_port)
  
    # reading the input using the camera
    result, image = cam.read()
  
    # If image will detected without any error, 
    # show result
    if result:
  
    # showing result, it take frame name and image 
    # output
        imshow("photo.jpg", image)
  
    # saving image in local storage
        imwrite("photo.jpg", image)
  
    # If keyboard interrupt occurs, destroy image 
    # window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
#        cv2.destroyWindow("photo.jpg")

  
# If captured image is corrupted, moving to else part
    else:
        print("No image detected. Please! try again")
       
        
  #with open(filename, 'wb') as f:
  #  f.write(binary)
    return image


# In[5]:


take_photo()


# In[6]:


#image = "photo.jpg"
#image = 'https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg'


from PIL import Image

#read the image
image = Image.open("photo.jpg")

#show image
#image.show()


# In[7]:


PPE(image)


# In[ ]:




