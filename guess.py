import os
import numpy as np
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

f1 = "test_images/my_ferrari.jpg"
f2 = "test_images/my_lamborghini.jpg"
f3 = "test_images/my_porche.jpg"
f4 = "test_images/another_lamborghini.jpg"
f5 = "test_images/mansion.jpg"

# Read test images using tensorflow
image_1 = tf.gfile.FastGFile(f1, 'rb').read()
image_2 = tf.gfile.FastGFile(f2, 'rb').read()
image_3 = tf.gfile.FastGFile(f3, 'rb').read()
image_4 = tf.gfile.FastGFile(f4, 'rb').read()
image_5 = tf.gfile.FastGFile(f5, 'rb').read()

images = []
images.append(image_1)
images.append(image_2)
images.append(image_3)
images.append(image_4)
images.append(image_5)

# Loads label file, strips off carriage return
label_porche = "Porche"
label_lamborghini = "Lamborghini"
label_ferrari = "Ferrari"

# Read the trained Model File
trained_model_file = tf.gfile.FastGFile("tf_files/retrained_graph.pb", 'rb').read()
graph_def = tf.GraphDef()

# Parse the Model File
graph_def.ParseFromString(trained_model_file)
tf.import_graph_def(graph_def, name='')


def predict_what_it_is(sess, image, softmax_tensor):
    prediction = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image})
    result = prediction[0]
    #Sort the labels in order of confidence 
    """result = predictions[0]
    result.sort()
    print(result)"""
    if(result[0] > 0.5):
        print(label_porche)
    elif(result[1] > 0.5):
        print(label_lamborghini)
    elif(result[2] > 0.5):
        print(label_ferrari)
    else:
        print("Cant Determine")


# Start a Tensorflow session
with tf.Session() as sess:
    # Feed the image_data as input to graph
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    # Predict the Object
    for file in images:
        predict_what_it_is(sess, file, softmax_tensor)
    
    
    
    
        
    
    
    
    
