import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#1 step of neural network creation of dataset
mnist=tf.keras.datasets.mnist

#2 step segregate the dataset and training image and label and testing images and labels
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

#3 step create the class name
class_names=["zero","one","two","three","four","five","six","seven","eight","nine","ten"]
#4 step display the image using plt.imshow,it displays ths first trining image
#plt.imshow(train_images[5])
#plt.show()

#step 5 train the images 
train_images=train_images/255.0
test_images=test_images/255.0

#step 6 create a model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation="softmax")
])
#first one is keras layes(input layer)
#second one is dense layes and in dense layer(hdden layer) put the activaton function means how can you display it
#third layer is Dense layer (output layer)
                                                                                                                                                  
#step 7 supply this architecture into the above  model
#model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accurecy"])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#loss is come under cast function
#adam is a type of optimizer

#step 8 here we compare the output with the image labels ,epochs is nothing but it is iteration,it gives loss and accuracy
model.fit(train_images,train_labels,epochs=5)

#step 9 here we test or evaluating the model
test_loss,test_acc=model.evaluate(test_images,test_labels)

#step 10 we can check the accuracy,if our accuracy is more than 90 then we start predicting
prediction=model.predict(test_images)

#here we predict  5 images , here cm.binary gives something change color from black and white
for i in range(5):
#plt.grid(False)
    plt.grid(False)
    plt.imshow(test_images[i],cmap=plt.cm.binary)
    plt.xlabel("Actual:"+class_names[test_labels[i]])
    plt.title("Prediction :" + class_names[np.argmax(prediction[i])])
    #plt.title("Prediction :"+class_names(np.argmax(prediction[i])])
    plt.show()
