print('Setting up')

# required libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='3'   # hide TF warnings, they get noisy
import socketio
import eventlet
import numpy as np 
from flask import Flask
from tensorflow import keras
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

# socket + flask setup
sio = socketio.Server()
app = Flask(__name__)
maxSpeed = 10   # cap speed so the car doesn’t go crazy

# quick preprocessing for images coming from simulator
def preProcess(img):
    img = img[60:135,:,:]                       # cut off sky and hood
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # YUV works better for driving tasks
    img = cv2.GaussianBlur(img, (3,3), 0)       # smooth out noise
    img = cv2.resize(img, (200,66))             # resize to model input
    img = img/255                               # normalize to 0–1
    return img

# this runs whenever the simulator sends us data
@sio.on('telementry')
def telementry(sid, data):
    speed = float(data['speed'])
    
    # decode the image from the simulator
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)
    
    # preprocess and make it batch‑like
    image = preProcess(image)
    image = np.array([image]) 
    
    # model predicts steering angle
    steering = float(model.predict(image))
    
    # throttle logic: slow down as we approach maxSpeed
    throttle = 1.0 - speed / maxSpeed
    
    print('{} {} {}'.format(steering, throttle, speed))
    
    # send steering + throttle back to sim
    sendControl(steering, throttle)

# when the simulator first connects
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)   # start with no movement

# helper to send commands back to the sim
def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering),
        'throttle': str(throttle)
    })

# entry point
if __name__ == '__main__':
    model = load_model('model.h5')   # load trained CNN
    app = socketio.Middleware(sio, app)
    # spin up server on port 4567
    eventlet.wsgi.server(eventlet.listen(('',4567)), app)
