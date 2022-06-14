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

import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d

import calendar
import time
from os.path import exists
import math
import numpy as np
import pandas as pd
import csv
if __name__ == '__main__':
  
    # images to be shown
    image_list = list()

    pathInPut ='E:/GRADUATE PROJECT/handpose_data.csv'
    pathImageFolder ='E:/GRADUATE PROJECT/images'
    
    pathOutPut ='E:/GRADUATE PROJECT/final_csv/All_Hand_Point_Angle.csv'
    pathOutPutImageFolder ='E:/GRADUATE PROJECT/check'
    
    
    csvfile = open(pathOutPut , 'w', newline='')
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['ImageName','X1','Y1','X2','Y2','X3','Y3','X4','Y4', 'Angle 1' ,'X5','Y5','X6','Y6','X7','Y7','X8','Y8','Angle 2','X9','Y9','X10','Y10','X11','Y11','X12','Y12','Angle 3','X13','Y13','X14','Y14','X15','Y15','X16','Y16','Angle 4','X17','Y17','X18','Y18','X19','Y19','X20','Y20'  ])

    col_list = ["ImageName","Angle1","Angle2","Angle3","Angle4"]
    data = pd.read_csv(pathInPut , header=0 , usecols =col_list)
    cols = data.shape[1]
    ImageName = data.iloc[0:,0:1]
    ImageAngle = data.iloc[0:,1:]
    images = np.array(ImageName.values) 
    Angles = np.array(ImageAngle.values)

    for image in images :
        imagename = str(image[:][0])
        imagePath =pathImageFolder+'/{}'.format(imagename)
        image_list.append(imagePath)
    
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

    index = 0
    # Feed image list through network
    for img_name in image_list:
    
      image_raw = scipy.misc.imread(img_name)
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

      # visualize
      fig = plt.figure(1)
      ax1 = fig.add_subplot(221)
      ax2 = fig.add_subplot(222)
      ax3 = fig.add_subplot(223)
      ax4 = fig.add_subplot(224,projection='3d')
      ax1.imshow(image_raw)
      plot_hand(coord_hw, ax1)
      ax2.imshow(image_crop_v)
      plot_hand(coord_hw_crop, ax2)
      ax3.imshow(np.argmax(hand_scoremap_v, 2))
      plot_hand_3d(keypoint_coord3d_v, ax4)
      ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
      ax4.set_xlim([-3, 3])
      ax4.set_ylim([-3, 1])
      ax4.set_zlim([-3, 3])

      imageName = images[index][0]

      # Save Photo
      plt.savefig(pathOutPutImageFolder+'/{}'.format(imageName))
      plt.clf()
      #plt.show()

      ##########################################################################  
      
      keypoint_coord3d_v = keypoint_coord3d_v.tolist()
      
      finger1 = keypoint_coord3d_v[1:5][:]

      finger2 = keypoint_coord3d_v[5:9][:]

      finger3 = keypoint_coord3d_v[9:13][:]

      finger4 = keypoint_coord3d_v[13:17][:]

      finger5 = keypoint_coord3d_v[17:][:]

      spamwriter.writerow([
        imageName,
        finger1[0][0],finger1[0][1],finger1[1][0],finger1[1][1],finger1[2][0],finger1[2][1],finger1[3][0],finger1[3][1],
        Angles[index][0],
        finger2[0][0],finger2[0][1],finger2[1][0],finger2[1][1],finger2[2][0],finger2[2][1],finger2[3][0],finger2[3][1],
        Angles[index][1],
        finger3[0][0],finger3[0][1],finger3[1][0],finger3[1][1],finger3[2][0],finger3[2][1],finger3[3][0],finger3[3][1],
        Angles[index][2],
        finger4[0][0],finger4[0][1],finger4[1][0],finger4[1][1],finger4[2][0],finger4[2][1],finger4[3][0],finger4[3][1],
        Angles[index][3],
        finger5[0][0],finger5[0][1],finger5[1][0],finger5[1][1],finger5[2][0],finger5[2][1],finger5[3][0],finger5[3][1] 
        ])
 
      index+=1
    csvfile.close()
