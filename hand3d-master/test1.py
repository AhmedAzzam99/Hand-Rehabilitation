#
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function, unicode_literals
import base64

from flask import Flask
from flask import request
import tensorflow as tf
import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy
import pandas as pd
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d
from sklearn import model_selection

import pickle
"""
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get('/')
def hello():
    print(__name__)
    return "Hello Abd Elrhman"
@app.post('/image')
def hello_Image():
    value = str(request.form['value'])
    base64_img_bytes = value.encode('utf-8')
    #print(base64_img_bytes)
    with open('image.png', 'wb') as file_to_save:
        decoded_image_data = base64.decodebytes(base64_img_bytes)
        file_to_save.write(decoded_image_data)
    return "ok"
    """
app = Flask(__name__)

@app.route('/')
def hello():
    print(__name__)
    return "Hello Abd Elrhman"
    
@app.route('/image', methods=["POST"])
def hello_Image():
    value = str(request.form['value'])
    base64_img_bytes = value.encode('utf-8')
    #print(base64_img_bytes)
    with open('ask.png', 'wb') as file_to_save:
        decoded_image_data = base64.decodebytes(base64_img_bytes)
        file_to_save.write(decoded_image_data)
        filepath = os.path.abspath(file_to_save.name)
        print(filepath)
   

      # images to be shown
    image_list='C:/Users/Ahmed Atef/ask.png'
    #image_list='C:/Users/User/Desktop/project/input/l2.jpg'

    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
    hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
    evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = ColorHandPose3DNetwork()
    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
    keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    net.init(sess)

    # Feed image list through network
    image_raw = scipy.misc.imread(image_list)
    image_raw = scipy.misc.imresize(image_raw, (240, 320))
    image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

    hand_scoremap_v, image_crop_v, scale_v, center_v,\
    keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,
                                                         keypoints_scoremap_tf, keypoint_coord3d_tf],
                                                        feed_dict={image_tf: image_v})

    hand_scoremap_v = np.squeeze(hand_scoremap_v)
    image_crop_v = np.squeeze(image_crop_v)
    keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
    keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

    # post processing
    image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
    coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)
    keypoint_coord3d_v = keypoint_coord3d_v.tolist()
    #divide data to fingers
    finger1 = keypoint_coord3d_v[1:5][:]
    finger2 = keypoint_coord3d_v[5:9][:]
    finger3 = keypoint_coord3d_v[9:13][:]
    finger4 = keypoint_coord3d_v[13:17][:]
    finger5 = keypoint_coord3d_v[17:][:]
    #use machine model to predict
    hand_model = pickle.load(open('G:/project/hand3d-master/find_hand2.pkl', 'rb'))
    hand2dpoints=[[finger1[0][0], finger1[0][1], finger1[1][0], finger1[1][1], finger1[2][0], finger1[2][1], finger1[3][0],finger1[3][1],finger2[0][0], finger2[0][1], finger2[1][0], finger2[1][1], finger2[2][0], finger2[2][1], finger2[3][0],finger2[3][1]]]
    lr=hand_model.predict(hand2dpoints)
    loaded_model = pickle.load(open('G:/project/hand3d-master/find_angle.pkl', 'rb'))
    an=[None,None,None,None]
    an[0]=[[finger1[0][0], finger1[0][1], finger1[1][0], finger1[1][1], finger1[2][0], finger1[2][1], finger1[3][0],finger1[3][1],finger2[0][0], finger2[0][1], finger2[1][0], finger2[1][1], finger2[2][0], finger2[2][1], finger2[3][0],finger2[3][1],1,lr]]
    an[1]=[[finger2[0][0], finger2[0][1], finger2[1][0], finger2[1][1], finger2[2][0], finger2[2][1], finger2[3][0],finger2[3][1],finger3[0][0], finger3[0][1], finger3[1][0], finger3[1][1], finger3[2][0], finger3[2][1], finger3[3][0],finger3[3][1],2,lr]]
    an[2]=[[finger3[0][0], finger3[0][1], finger3[1][0], finger3[1][1], finger3[2][0], finger3[2][1], finger3[3][0],finger3[3][1],finger4[0][0], finger4[0][1], finger4[1][0], finger4[1][1], finger4[2][0], finger4[2][1], finger4[3][0],finger4[3][1],3,lr]]
    an[3]=[[finger4[0][0], finger4[0][1], finger4[1][0], finger4[1][1], finger4[2][0], finger4[2][1], finger4[3][0],finger4[3][1],finger5[0][0], finger5[0][1], finger5[1][0], finger5[1][1], finger5[2][0], finger5[2][1], finger5[3][0],finger5[3][1],4,lr]]
    prediction=[None,None,None,None]
    result= ""
    for i in range (4):
        prediction[i] = loaded_model.predict(an[i])
        result+=str(loaded_model.predict(an[i]))+ "#"
    print(prediction[0])
    print(prediction[1])
    print(prediction[2])
    print(prediction[3])
    
    return str(result)
    
    
if __name__ == '__main__':
    app.run(host="0.0.0.0",debug= True)
  
