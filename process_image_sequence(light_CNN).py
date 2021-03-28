import numpy as np
import matplotlib.pyplot as plt    
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.applications.vgg16 import preprocess_input


# =============================================================================
# # The video with infiltration
# csv_file_test='/home/jianxig/test_set_17_videos/80/labels(80).csv'
# test_data_dir = "/home/jianxig/test_set_17_videos/80/frames/"
# nb_test_samples=602
# =============================================================================

# The video without infiltration
csv_file_test='/home/jianxig/test_set_17_videos/151/labels(151).csv'
test_data_dir = "/home/jianxig/test_set_17_videos/151/frames/"
nb_test_samples=476

img_width = 112
img_height = 112
batch_size = 1 # The batch size should be 1, which simplifys the procedure of getting the labels. 

# Load the filenames and labels
dataFrame=pd.read_csv(csv_file_test, delimiter=',')
dataFrame.Labels=dataFrame.Labels.astype(str) # Change the datatype of the column 'Labels'

# Preprocess the images
datagen = ImageDataGenerator(preprocessing_function=preprocess_input) 

generator_test= datagen.flow_from_dataframe(
        dataFrame,    
        directory = test_data_dir,    
        x_col = 'File_Name',    
        y_col = 'Labels',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False) 

#%% Construct the customized network
model=Sequential()

# The first CNN block
model.add(
    Conv2D(input_shape=(112,112,3), filters=16, kernel_size=(3,3), strides=(1, 1), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# The seccond CNN block
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# The third CNN block
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# Convert the 2D features to the 1D features
model.add(Flatten())

# The output layer
model.add(Dense(3, activation='softmax'))

# Load the optimal weights
model.load_weights('/home/jianxig/CNN_17/dataset_17/16_32_32/bestWeights_1_aug_Jan_22.hdf5')

# Configure the learning process of the neural network
model.compile(optimizer=Adam(lr = 0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
 
# (For the videos without infiltration) Regenerate the labels
i=0
test_label=[]
for i in range(len(dataFrame)):
    if dataFrame.iat[i,1]=='0':
        test_label.extend(np.array([[1., 0., 0.]]))
    if dataFrame.iat[i,1]=='1':
        test_label.extend(np.array([[0., 1., 0.]]))
    if dataFrame.iat[i,1]=='2':
        test_label.extend(np.array([[0., 0., 1.]]))

np_label=np.array(test_label)        
        
# (For the videos without infiltration) Extract the frames from the image data generator        
i=0
list_frame=[]
for i in range(len(generator_test)):
    list_frame.append(generator_test[i][0][0])    
    
np_frame=np.array(list_frame)
    
# (For the videos without infiltration) Evaluate the network by the test set
model.evaluate(np_frame, np_label, steps = nb_test_samples/batch_size, verbose=1)

# (For the videos without infiltration) Make predictions for the test images
predictions = model.predict(np_frame, steps = nb_test_samples/batch_size, verbose=1)   
     
# =============================================================================
# # (For the videos with infiltration) Evaluate the network by the test set
# model.evaluate(generator_test, steps = nb_test_samples/batch_size ,verbose=1) 
# 
# # (For the videos with infiltration) Make predictions for the test images
# predictions = model.predict(generator_test, steps = nb_test_samples/batch_size, verbose=1) 
# =============================================================================

# Generate the ground truth 
ground_truth=[]
j=0
for j in range(nb_test_samples):
      Label_0=generator_test[j][1][0]
      single_ground_truth = np.argmax(Label_0)
      ground_truth.append(single_ground_truth)

# Generate the prediction results
prediction_result=[]
k=0
for k in range(len(predictions)):
    pred = np.argmax(predictions[k])
    prediction_result.append(pred)

# Plot the ground truth and the prediction results
plt.figure(1)
plt.plot(ground_truth)
plt.plot(prediction_result)
plt.title('The performance of the network')
plt.ylabel('status')
plt.xlabel('frame number')
plt.legend(['Ground truth', 'predictions'], loc='upper left')
plt.show()

#%% Save the prediction result
prediction_txt_n='/home/jianxig/CNN_17/dataset_17/16_32_32/seq_results/151.txt'

with open(prediction_txt_n, 'w') as f:
    for item in prediction_result:
        f.write('{}\n'.format(item))
