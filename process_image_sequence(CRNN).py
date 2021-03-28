import numpy as np
import matplotlib.pyplot as plt    
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, TimeDistributed,LSTM
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
timesteps_for_LSTM = 30

# Load the filenames and labels
dataFrame=pd.read_csv(csv_file_test, delimiter=',')
dataFrame.Labels=dataFrame.Labels.astype(str) 

# Generate the labels
test_labels_df=[]    
for i in range(len(dataFrame)):
    if dataFrame.iat[i, 1] == '0':
            test_labels_df.extend(np.array([[1., 0., 0.]]))
    if dataFrame.iat[i, 1] == '1':
            test_labels_df.extend(np.array([[0., 1., 0.]]))
    if dataFrame.iat[i, 1] == '2':
            test_labels_df.extend(np.array([[0., 0., 1.]]))
test_labels_df_1=np.array(test_labels_df)

# Preprocess the images, but avoid data augmentation
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

# Extract processed frames and labels from image_data_generator
i=0
list_label=[]
list_frame=[]
for i in range(len(generator_test)):
    list_frame.append(generator_test[i][0][0])
    list_label.append(generator_test[i][1][0])    
np_frame=np.array(list_frame)
np_label=np.array(list_label)

# Modify the list of label
# Insert an extra label at the begining of label list
np_label_new = np.insert(test_labels_df_1, 0, [1.,0.,0.], axis=0)

# Remove the last element within label list
np_label_new_1 = np_label_new[:-1]

# Generate time series data
generator_test_time = TimeseriesGenerator(np_frame, np_label_new_1, length = timesteps_for_LSTM, batch_size = 1)

# Construct the cnn model
cnn_model=Sequential()

# The first CNN block
cnn_model.add(
    Conv2D(input_shape=(112,112,3), filters=16, kernel_size=(3,3), strides=(1, 1), padding="same", activation="relu"))

cnn_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# The seccond CNN block
cnn_model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding="same", activation="relu"))

cnn_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# The third CNN block
cnn_model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding="same", activation="relu"))

cnn_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# Convert the 2D features to the 1D features
cnn_model.add(Flatten())

# The output layer
cnn_model.add(Dense(3, activation='softmax'))

# Load the pre-trained weights of the cnn model
cnn_model.load_weights('/home/jianxig/CNN_17/dataset_17/16_32_32/bestWeights_1_aug_Jan_22.hdf5')

# Replace the output layer in the cnn model with a 32-neuron dense layer
dense_1=Dense(32, activation='relu')(cnn_model.layers[-2].output)

# Define the modified CNN model
modified_cnn=Model(inputs = cnn_model.inputs, outputs=dense_1)

# Freeze the weights within the cnn layers
layer=0
for layer in modified_cnn.layers[:-1]:
    layer.trainable = False
    
# DIsplay the modified CNN model    
modified_cnn.summary()

# Define the CRNN model
model=Sequential()

# Add the cnn model
model.add(TimeDistributed(modified_cnn, input_shape=(timesteps_for_LSTM,112,112,3)))

# The LSTM layer.
model.add(LSTM(32, dropout=0.5))

# The output layer
model.add(Dense(3, activation='softmax'))

# Display the architecture of CRNN
model.summary()

# Load the optimal weights
model.load_weights('/home/jianxig/CNN_LSTM_17/30_timesteps/bestWeights_0_aug_w_weights_Jan_23.hdf5')

# Configure the learning process of the neural network
model.compile(optimizer=Adam(lr = 0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluate the network by the test set
model.evaluate(generator_test_time, steps = len(generator_test_time)/batch_size ,verbose=1) 

# Make predictions for the test images
predictions = model.predict(generator_test_time, steps = len(generator_test_time)/batch_size, verbose=1) 

# Generate the ground truth 
ground_truth=[]
j=0
for j in range(len(generator_test_time)):
      Label_0=generator_test_time[j][1][0]
      single_ground_truth = np.argmax(Label_0)
      ground_truth.append(single_ground_truth)

# Generate the prediction results
prediction_result=[]
k=0
for k in range(len(predictions)):
    pred = np.argmax(predictions[k])
    prediction_result.append(pred)
  
#%% Plot the ground truth and the prediction results
plt.figure(1)
plt.plot(ground_truth)
plt.plot(prediction_result)
plt.title('The performance of the network')
plt.ylabel('status')
plt.xlabel('frame number')
plt.legend(['Ground truth', 'predictions'], loc='upper left')
plt.show()

#%% Save the prediction results
prediction_txt_n='/home/jianxig/CNN_LSTM_17/30_timesteps/seq_data/151.txt'

with open(prediction_txt_n, 'w') as f:
    for item in prediction_result:
        f.write('{}\n'.format(item))
