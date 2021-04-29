import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt    
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, TimeDistributed,LSTM 

csv_file_test='/home/jianxig/test_set_17_combo/labels_for_test_seq(d_17).csv'
test_data_dir = "/home/jianxig/test_set_17_combo/frames/"
img_width = 112
img_height = 112
batch_size = 1 # The batch size should be 1, which simplifys the procedure of getting the labels. 
nb_test_samples = 6190
timesteps_for_LSTM = 30

# Load the filenames and labels
dataFrame=pd.read_csv(csv_file_test, delimiter=',')
dataFrame.Labels=dataFrame.Labels.astype(str) 

# Generate the labels
testing_labels_df=[]    
for i in range(len(dataFrame)):
    if dataFrame.iat[i, 1] == '0':
            testing_labels_df.extend(np.array([[1., 0., 0.]]))
    if dataFrame.iat[i, 1] == '1':
            testing_labels_df.extend(np.array([[0., 1., 0.]]))
    if dataFrame.iat[i, 1] == '2':
            testing_labels_df.extend(np.array([[0., 0., 1.]]))
testing_labels_df_1=np.array(testing_labels_df)

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
np_label_new = np.insert(np_label, 0, [1.,0.,0.], axis=0)

# Remove the last element within label list
np_label_new_1 = np_label_new[:-1]

# Generate time series data
generator_test_time = TimeseriesGenerator(np_frame, np_label_new_1, length = timesteps_for_LSTM, batch_size = 1)

# Construct the network
cnn_model=Sequential()

# The first CNN block
cnn_model.add(
    Conv2D(input_shape=(112,112,3), filters=16, kernel_size=(3,3), strides=(1, 1), padding="same", activation="relu"))

cnn_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# The second CNN block
cnn_model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding="same", activation="relu"))

cnn_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# The third CNN block
cnn_model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding="same", activation="relu"))

cnn_model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# Convert the 2D features to the 1D features
cnn_model.add(Flatten())

# The output layer
cnn_model.add(Dense(3, activation='softmax'))

# Load the optimal weights for the cnn 
cnn_model.load_weights('/home/jianxig/CNN_17/dataset_17/16_32_32/bestWeights_1_aug_Jan_22.hdf5')

# Replace the output layer in the cnn model with a 32-neuron dense layer
dense_1=Dense(32, activation='relu')(cnn_model.layers[-2].output)

# Define the modified cnn model
modified_cnn=Model(inputs = cnn_model.inputs, outputs=dense_1)

# Display the cnn model
modified_cnn.summary()

# Freeze the weights within the cnn layers
layer=0
for layer in modified_cnn.layers[:-1]:
    layer.trainable = False
    
# Define the CRNN model
model=Sequential()

# Add the modified cnn model
model.add(TimeDistributed(modified_cnn, input_shape=(timesteps_for_LSTM,112,112,3)))

# The LSTM layer.
model.add(LSTM(32, dropout=0.5))

# The output layer
model.add(Dense(3, activation='softmax'))

# Load the optimal weights for the LSTM 
model.load_weights('/home/jianxig/CNN_LSTM_17/30_timesteps/bestWeights_0_aug_w_weights_Jan_23.hdf5')

# Display the CRNN
model.summary()

# Configure the learning process of the neural network
model.compile(optimizer=Adam(lr = 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Evaluate the network by the test set
model.evaluate(generator_test_time, steps = len(generator_test_time)/batch_size ,verbose=1) 

# Make predictions for the test images
predictions = model.predict(generator_test_time, steps = len(generator_test_time)/batch_size, verbose=1) 

# Generate the prediction results
predicted_result = np.argmax(predictions, axis=1)

# Extract labels from generator_test_time
true_label=[]
j=0
for j in range(len(generator_test_time)):
    Label_0=generator_test_time[j][1][0]
    single_true_label=np.argmax(Label_0)
    true_label.append(single_true_label)
true_label=np.asarray(true_label)

# Generate the confusion matrix
confusionMat=confusion_matrix(
    true_label, predicted_result)

# Set font format
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Force matplotlib to use TrueType font
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

category=['No needle', 'In fistula', 'Infiltration']
fig, ax = plt.subplots()
im = ax.imshow(confusionMat,cmap = 'Reds')
ax.set_xticks(np.arange(len(category)))
ax.set_yticks(np.arange(len(category)))
ax.set_xticklabels(category)
ax.set_yticklabels(category)
plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
         rotation_mode="anchor")
plt.setp(ax.get_yticklabels(), rotation=90, ha="center",
         rotation_mode="anchor")
ax.tick_params(axis='both', which='major', labelsize=16)
ax.tick_params(axis='y', which='major', pad=10)

# Show the results in percentage
confusionMat_percent = np.zeros((3,3))
number_of_frames=[(3903-timesteps_for_LSTM), 1117, 1170]
for i in range(len(category)):
    for j in range(len(category)):
        confusionMat_percent[i,j] = confusionMat[i,j]/number_of_frames[i]

# Round the results
CM_percent_round = np.zeros((3,3))
np.round(confusionMat_percent,decimals=2,out=CM_percent_round)

# Loop over data dimensions and create text annotations. 
for i in range(len(category)):
    for j in range(len(category)):
        if (i==0 and j==0):
            text = ax.text(j, i, CM_percent_round[i, j], ha="center", va="center", color="w", fontsize=20)
        else:
            text = ax.text(j, i, CM_percent_round[i, j], ha="center", va="center", color="k", fontsize=20)
