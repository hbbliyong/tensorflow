import tensorflow as tf
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

file_path="img"
filename="./img/121.jpg"

raw_image_data=cv2.imread(filename)
image=tf.placeholder('uint8',[None,None,3])
_slice=tf.slice(image,[100,0],[100,-1])

with tf.Session() as session:
    result=session.run(_slice,feed_dict={image:raw_image_data})
    print(result.shape)

cv2.namedWindow('result',0)
cv2.imshow('result',result)
cv2.waitKey(0)    