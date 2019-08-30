import cv2
import numpy as np
import os

from sklearn.preprocessing import label_binarize #For creating one hot labels

from keras import models, layers, optimizers #Importing modules from Keras that will help us create a model

# from keras.applications.mobilenet_v2 import MobileNetV2 as mblv2 #Importing the model

#Methods of MobileNet for making changing the input to the required shape and then decoding the output
from keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from keras.models import load_model

image_rows = 64
image_columns = 64
filter_size = 3
classesReal=["0","none","8","none","a","none","b","none","c","none","f","none","k","none","m","none"]
classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

all_data = np.zeros((800,64,64,1))
all_labels =np.zeros(800)
def reading_data_and_labels():
    """
    This function is intended to pre process the data so that it can be fed into the network for training
    """
    global all_data
    global all_labels

    data_folder0 = 'E:/random/AliImages/american-sign-language-dataset/asl/0'
    data_folder8 = 'E:/random/AliImages/american-sign-language-dataset/asl/8'
    data_foldera = 'E:/random/AliImages/american-sign-language-dataset/asl/a'
    data_folderb = 'E:/random/AliImages/american-sign-language-dataset/asl/b'
    data_folderc = 'E:/random/AliImages/american-sign-language-dataset/asl/c'
    data_folderf = 'E:/random/AliImages/american-sign-language-dataset/asl/f'
    data_folderk = 'E:/random/AliImages/american-sign-language-dataset/asl/k'
    data_folderm = 'E:/random/AliImages/american-sign-language-dataset/asl/m'

    # path_to_label_file = 'E:/random/AliImages/Data/labels.csv'

    for i in range(1,800,8):
        for counter,filename in enumerate(os.listdir(data_folder0)):

            img = cv2.imread(os.path.join(data_folder0, filename),0)

            img = cv2.resize(img, (64,64))
            all_labels[i]=classes[1]
            if img is not None:
                all_data[i,:,:,0] = img

    for i in range(2,800,8):
        for counter,filename in enumerate(os.listdir(data_folder8)):

            img = cv2.imread(os.path.join(data_folder8, filename),0)

            img = cv2.resize(img, (64,64))
            all_labels[i]=classes[3]
            if img is not None:
                all_data[i,:,:,0] = img


    for i in range(3,800,8):
        for counter,filename in enumerate(os.listdir(data_foldera)):

            img = cv2.imread(os.path.join(data_foldera, filename),0)

            img = cv2.resize(img, (64,64))
            all_labels[i]=classes[5]
            if img is not None:
                all_data[i,:,:,0] = img

    for i in range(4,800,8):
        for counter,filename in enumerate(os.listdir(data_folderb)):

            img = cv2.imread(os.path.join(data_folderb, filename),0)

            img = cv2.resize(img, (64,64))
            all_labels[i]=classes[7]
            if img is not None:
                all_data[i,:,:,0] = img

    for i in range(5,800,8):
        for counter,filename in enumerate(os.listdir(data_folderc)):

            img = cv2.imread(os.path.join(data_folderc, filename),0)

            img = cv2.resize(img, (64,64))
            all_labels[i]=classes[9]
            if img is not None:
                all_data[i,:,:,0] = img

    for i in range(6,800,8):
        for counter,filename in enumerate(os.listdir(data_folderf)):

            img = cv2.imread(os.path.join(data_folderf, filename),0)

            img = cv2.resize(img, (64,64))
            all_labels[i]=classes[11]
            if img is not None:
                all_data[i,:,:,0] = img

    for i in range(7,800,8):
        for counter,filename in enumerate(os.listdir(data_folderk)):

            img = cv2.imread(os.path.join(data_folderk, filename),0)

            img = cv2.resize(img, (64,64))
            all_labels[i]=classes[13]
            if img is not None:
                all_data[i,:,:,0] = img

    for i in range(8,800,8):
        for counter,filename in enumerate(os.listdir(data_folderm)):

            img = cv2.imread(os.path.join(data_folderm, filename),0)

            img = cv2.resize(img, (64,64))
            all_labels[i]=classes[15]
            if img is not None:
                all_data[i,:,:,0] = img



    # all_labels = np.loadtxt(path_to_label_file)
    # all_labels=all_labels[0:250]
    all_labels = label_binarize(all_labels, classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

    print("Done")


def main():

    global all_data
    global all_labels
    reading_data_and_labels()









    #Adding layers to a model
    my_model = models.Sequential()
    # data_folder = 'E:/random/AliImages/american-sign-language-dataset/asl/0'




     # checking if checkpoint saved, if not, training and saving the checkpoint,EDITED BY SAQIBBB
    if(os.path.exists("./ckpt2/model")):
        my_model=load_model("./ckpt2/model")



    #
    # #Each block will look something like this:
    # my_model.add(layers.Conv2D(8, (4,4), padding= "SAME", activation='relu',
    #                            input_shape=(image_rows, image_columns, 1)))
    # my_model.add(layers.MaxPooling2D(pool_size=(8,8), strides=(8,8)))
    #
    # my_model.add(layers.Conv2D(16, (2, 2), padding="SAME", activation='relu'))
    # my_model.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))
    #
    # my_model.add(layers.Flatten())
    #
    # my_model.add(layers.Dense(16, activation='softmax'))
    #
    # #Displaying the summary of the model to check whether it is what we want
        my_model.summary()

        #Compiling the model
        my_model.compile(loss="categorical_crossentropy", optimizer = optimizers.Adam(lr=0.001), metrics=['acc'])


        #Training the model
        my_history = my_model.fit(all_data[0:800,:,:,:], all_labels[0:800,:], batch_size=8, epochs=15, validation_split=0.2)
        my_model.save("E:/random/TF_Keras/TF & Keras/ckpt2/model.")

        #Making predictions on the test data
        my_predictions = my_model.predict(all_data[200:250,:,:,:])
        print("the preds are here")
        print(my_predictions)

    #Evaluating loss and other matrics
    # my_loss_and_metrics = my_model.evaluate(all_data[120:151,:,:,:], all_labels[120:151,:], batch_size=16)
    #print(my_loss_and_metrics)



main()
