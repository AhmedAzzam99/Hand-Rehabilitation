
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
    #imageName_list = list()

    #image_list.append('./data/0210.jpg')
    #image_list.append('/content/drive/MyDrive/data/aaa.jpg')

    pathInPut ='G:/project/Hand_Angle4.csv'
    pathOutPut ='G:/project/data_point_angle.csv'
    #/content/drive/MyDrive/ResultOutputAngle/All_Hand_Angle.csv
    csvfile = open(pathOutPut , 'w', newline='')
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar=',', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['ImageName','X1','Y1','X2','Y2','X3','Y3','X4','Y4', 'Angle' ,'X5','Y5','X6','Y6','X7','Y7','X8','Y8' ])

    col_list = ["ImageName","Angle1","Angle2","Angle3","Angle4"]
    data = pd.read_csv(pathInPut , header=0 , usecols =col_list)
    cols = data.shape[1]
    ImageName = data.iloc[0:,0:1]
    ImageAngle = data.iloc[0:,1:]   
    Images = np.array(ImageName.values) 
    Angles = np.array(ImageAngle.values)

    for image in Images :
        imagename = str(image[:][0])
        #imageName_list.append(imagename)
        imagePath ='G:/project/gehad1/{}'.format(imagename)
        image_list.append(imagePath)
    
    # network input
    # New Edite
    #image_tf =tf.disable_v2_behavior(tf.float32, shape=(1, 240, 320, 3))
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

    #exclFile = open("./drive/MyDrive/result/output.txt",'w')
    
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

      '''
      if keypoint_coord3d_v[0][1] > keypoint_coord3d_v[1][1]:
        keypoint_coord3d_v[0][1] += 0.5
      else:
        keypoint_coord3d_v[0][1]=keypoint_coord3d_v[0][1] - 0.5
      '''

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

      #imageName = imageName_list[index]
      imageName = Images[index][0]
      #outPutPath = "/content/drive/MyDrive/outputDataModel/{}".format(indexName+1)
      
      #exclFile = open(outPutPath+'.txt','w')

      # Save Photo
      #imageName = str(calendar.timegm(time.gmtime()))
      #plt.savefig('./drive/MyDrive/result/{}'.format(imageName,'.png'))
      plt.savefig('G:/project/gehadoutput/{}'.format(imageName))
      plt.clf()
      ##########################################################################  
      
      keypoint_coord3d_v = keypoint_coord3d_v.tolist()
      
      finger1 = keypoint_coord3d_v[1:5][:]
      #finger1.insert(0, keypoint_coord3d_v[0][:] )

      finger2 = keypoint_coord3d_v[5:9][:]
      #finger2.insert(0, keypoint_coord3d_v[0][:])

      finger3 = keypoint_coord3d_v[9:13][:]
      #finger3.insert(0, keypoint_coord3d_v[0][:])

      finger4 = keypoint_coord3d_v[13:17][:]
      #finger4.insert(0,keypoint_coord3d_v[0][:])

      finger5 = keypoint_coord3d_v[17:][:]
      #finger5.insert(0, keypoint_coord3d_v[0][:])

      # spamwriter.writerow([
      #   imageName,
      #   finger1[0][0],finger1[0][1],finger1[1][0],finger1[1][1],finger1[2][0],finger1[2][1],finger1[3][0],finger1[3][1],
      #   Angles[index][0],
      #   finger2[0][0],finger2[0][1],finger2[1][0],finger2[1][1],finger2[2][0],finger2[2][1],finger2[3][0],finger2[3][1],
      #   Angles[index][1],
      #   finger3[0][0],finger3[0][1],finger3[1][0],finger3[1][1],finger3[2][0],finger3[2][1],finger3[3][0],finger3[3][1],
      #   Angles[index][2],
      #   finger4[0][0],finger4[0][1],finger4[1][0],finger4[1][1],finger4[2][0],finger4[2][1],finger4[3][0],finger4[3][1],
      #   Angles[index][3],
      #   finger5[0][0],finger5[0][1],finger5[1][0],finger5[1][1],finger5[2][0],finger5[2][1],finger5[3][0],finger5[3][1]
      #   ])

      spamwriter.writerow([
        imageName,
        finger1[0][0], finger1[0][1], finger1[1][0], finger1[1][1], finger1[2][0], finger1[2][1], finger1[3][0],finger1[3][1],
        Angles[index][0],
        finger2[0][0], finger2[0][1], finger2[1][0], finger2[1][1], finger2[2][0], finger2[2][1], finger2[3][0],finger2[3][1]
      ])

      spamwriter.writerow([
        imageName,
        finger2[0][0], finger2[0][1], finger2[1][0], finger2[1][1], finger2[2][0], finger2[2][1], finger2[3][0],finger2[3][1],
        Angles[index][1],
        finger3[0][0], finger3[0][1], finger3[1][0], finger3[1][1], finger3[2][0], finger3[2][1], finger3[3][0],finger3[3][1]
      ])

      spamwriter.writerow([
        imageName,
        finger3[0][0], finger3[0][1], finger3[1][0], finger3[1][1], finger3[2][0], finger3[2][1], finger3[3][0],finger3[3][1],
        Angles[index][2],
        finger4[0][0], finger4[0][1], finger4[1][0], finger4[1][1], finger4[2][0], finger4[2][1], finger4[3][0],finger4[3][1]
      ])

      spamwriter.writerow([
        imageName,
        finger4[0][0], finger4[0][1], finger4[1][0], finger4[1][1], finger4[2][0], finger4[2][1], finger4[3][0],finger4[3][1],
        Angles[index][3],
        finger5[0][0], finger5[0][1], finger5[1][0], finger5[1][1], finger5[2][0], finger5[2][1], finger5[3][0],finger5[3][1]
      ])
 
      index+=1

      
      # Save Note Text
      exclFile.write("keypoint_coord3d_v \n")
      exclFile.write(str(keypoint_coord3d_v))
      exclFile.write("\n")
      exclFile.write("\n")
      '''
      '''
      exclFile.write("Finger List")
      exclFile.write("\n")
      exclFile.write(str(finger1))
      exclFile.write("\n")
      exclFile.write(str(finger2))
      exclFile.write("\n")
      exclFile.write(str(finger3))
      exclFile.write("\n")
      exclFile.write(str(finger4))
      exclFile.write("\n")
      exclFile.write(str(finger5))
      exclFile.write("\n")
      '''
      '''
      listPoint =[keypoint_coord3d_v[0][:] , keypoint_coord3d_v[1][:] , keypoint_coord3d_v[5][:] , keypoint_coord3d_v[9][:] , keypoint_coord3d_v[13][:] , keypoint_coord3d_v[17][:]] 

      for i,j in ((1,2) ,(2,3) ,(3,4),(4,5)):
        A=listPoint[0]
        B=listPoint[i]
              
        C=listPoint[0]
        D=listPoint[j]
        
        center = [0, 0, 0]
        fingerA = [B[0]-A[0],B[1]-A[1],B[2]-A[2]]
        fingerB = [D[0]-C[0],D[1]-C[1],D[2]-C[2]]
        
        FA_FB =fingerA[0]*fingerB[0] + fingerA[1]*fingerB[1]  + fingerA[2]*fingerB[2]
        lfA=math.sqrt(math.pow((fingerA[0]-center[0]), 2) + math.pow((fingerA[1]-center[1]), 2) + math.pow((fingerA[2]-center[2]), 2))
        lfB=math.sqrt(math.pow((fingerB[0]-center[0]), 2) + math.pow((fingerB[1]-center[1]), 2) + math.pow((fingerB[2]-center[2]), 2))
        
        r = FA_FB / (lfA * lfB)
        
        theta = math.degrees(math.acos(r))
        exclFile.write("Finger : ")
        exclFile.write(str(i))
        exclFile.write(" || Finger : ")
        exclFile.write(str(j))
        exclFile.write(" = ")
        exclFile.write(str(theta))
        exclFile.write("\n")
        
      #exclFile.close()
    csvfile.close()
