import tensorflow as tf
import cv2

file_path="img"
filename="./img/b.jpg"

image=cv2.imread(filename,1)
cv2.namedWindow('image',0)
cv2.imshow('image',image)

#Create a TensorFlow Variable
x=tf.Variable(image,name='x')

model=tf.initialize_all_variables()

with tf.Session() as session:
    x=tf.transpose(x,perm=[1,0,2])
    session.run(model)
    result=session.run(x)

cv2.namedWindow('result',0)
cv2.imshow('result',result)
cv2.waitKey(0)    