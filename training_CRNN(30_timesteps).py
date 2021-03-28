import json
import pandas as pd
import numpy as np
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, TimeDistributed,LSTM

img_width = 112
img_height = 112 
batch_size_for_imageDataGenerator = 1
time_steps_for_LSTM = 30
epochs=50
nb_train_samples=49962
csv_file_training = "/home/jianxig/training_set_17_combo/labels_for_training_seq(d_17).csv"
training_frames_dir = "/home/jianxig/training_set_17_combo/frames/"

# Load the filenames and labels
dataFrame=pd.read_csv(csv_file_training, delimiter=',')
dataFrame.Labels=dataFrame.Labels.astype(str)

# Generate the labels
train_labels_df=[]    
for i in range(len(dataFrame)):
    if dataFrame.iat[i, 1] == '0':
            train_labels_df.extend(np.array([[1., 0., 0.]]))
    if dataFrame.iat[i, 1] == '1':
            train_labels_df.extend(np.array([[0., 1., 0.]]))
    if dataFrame.iat[i, 1] == '2':
            train_labels_df.extend(np.array([[0., 0., 1.]]))
train_labels_df_1=np.array(train_labels_df)

# Preprocess the training set, but without data augmentation 
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Preprocessing the training set.
generator_train = datagen.flow_from_dataframe(
        dataFrame, 
        directory = training_frames_dir,
        x_col = 'File_Name', 
        y_col = 'Labels',
        target_size = (img_height, img_width),
        class_mode='categorical',
        batch_size=batch_size_for_imageDataGenerator,
        shuffle=False) 

# Calculate weights for unbalanced classes within training set
counter = Counter(dataFrame.iloc[:, 1].tolist()) # Count how many images within each class
max_val = float(max(counter.values()))
class_weights = {int(class_id,10) : max_val/num_images for class_id, num_images in counter.items()}

# Extract processed frames and labels from image_data_generator
i=0
list_label=[]
list_frame=[]
for i in range(len(generator_train)):
    list_frame.append(generator_train[i][0][0])
    list_label.append(generator_train[i][1][0])    
    
    # (For the purpose of debugging)
    if i%10000==0:
        print("I am generating the training data.")

np_frame=np.array(list_frame)
np_label=np.array(list_label)

# Modify the list of label
# Insert an extra label at the begining of label list
np_label_new = np.insert(np_label, 0, [1.,0.,0.], axis=0)

# Remove the last element within label list
np_label_new_1 = np_label_new[:-1]

# Generate time series data
generator_train_time = TimeseriesGenerator(np_frame, np_label_new_1, length = time_steps_for_LSTM, batch_size = 1)

# For the validation set
nb_validation_samples=5810
csv_file_validation = "/home/jianxig/validation_set_17_seq_combo/labels_for_validation_seq(d_17).csv"
validation_frames_dir = "/home/jianxig/validation_set_17_seq_combo/frames/"

# Load the filenames and labels
dataFrame_valid=pd.read_csv(csv_file_validation, delimiter=',')
dataFrame_valid.Labels=dataFrame_valid.Labels.astype(str) 

# Generate the labels
validation_labels_df=[]    
for i in range(len(dataFrame_valid)):
    if dataFrame_valid.iat[i, 1] == '0':
            validation_labels_df.extend(np.array([[1., 0., 0.]]))
    if dataFrame_valid.iat[i, 1] == '1':
            validation_labels_df.extend(np.array([[0., 1., 0.]]))
    if dataFrame_valid.iat[i, 1] == '2':
            validation_labels_df.extend(np.array([[0., 0., 1.]]))
validation_labels_df_1=np.array(validation_labels_df)

# Preprocess the validation set, but without data augmentation
datagen_valid = ImageDataGenerator(preprocessing_function=preprocess_input)

# Preprocessing the validation set.
generator_validation = datagen_valid.flow_from_dataframe(
        dataFrame_valid, 
        directory = validation_frames_dir,
        x_col = 'File_Name', 
        y_col = 'Labels',
        target_size = (img_height, img_width),
        class_mode='categorical',
        batch_size=batch_size_for_imageDataGenerator,
        shuffle=False) 

# Extract processed frames and labels from image_data_generator
i=0
list_label_v=[]
list_frame_v=[]
for i in range(len(generator_validation)):
    list_frame_v.append(generator_validation[i][0][0])
    list_label_v.append(generator_validation[i][1][0])    
    
    # (For the purpose of debugging)
    if i%1000==0:
        print("I am generating the validation data.")

np_frame_v=np.array(list_frame_v)
np_label_v=np.array(list_label_v)

# Modify the list of label
# Insert an extra label at the begining of label list
np_label_v_new = np.insert(np_label_v, 0, [1.,0.,0.], axis=0)

# Remove the last element within label list
np_label_v_new_1 = np_label_v_new[:-1]

# Generate time series data
generator_validation_time = TimeseriesGenerator(np_frame_v, np_label_v_new_1, length = time_steps_for_LSTM, batch_size = 1)

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

# Display the architecture
cnn_model.summary()

# Load the pre-trained weights of the cnn model
cnn_model.load_weights('/home/jianxig/CNN_17/dataset_17/16_32_32/bestWeights_1_aug_Jan_22.hdf5')

# Replace the output layer in the cnn model with a 32-neuron dense layer
dense_1=Dense(32, activation='relu')(cnn_model.layers[-2].output)

# Define the modified cnn model
modified_cnn=Model(inputs = cnn_model.inputs, outputs=dense_1)

# Display the modified cnn model
modified_cnn.summary()

# Freeze the weights within the cnn layers
layer=0
for layer in modified_cnn.layers[:-1]:
    layer.trainable = False
     
# Define the CRNN model
model=Sequential()

# Add the cnn model
model.add(TimeDistributed(modified_cnn, input_shape=(time_steps_for_LSTM,112,112,3)))

# The LSTM layer.
model.add(LSTM(32, dropout=0.5))

# The output layer
model.add(Dense(3, activation='softmax'))

# Display the architecture
model.summary()

# Save the architecture of the network
with open('/home/jianxig/CNN_LSTM_17/30_timesteps/arch_w_weights_Jan_23.json','w') as f:
    f.write(model.to_json())
    
# Configure the learning process of the customized network
model.compile(optimizer=Adam(lr = 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Save the optimal weights 
checkpointer = ModelCheckpoint(filepath='/home/jianxig/CNN_LSTM_17/30_timesteps/bestWeights_0_aug_w_weights_Jan_23.hdf5', 
                               monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')

# Train the customized network (weights are used) 
step_for_training = len(generator_train_time)
step_for_validation = len(generator_validation_time)
model_history = model.fit_generator(generator_train_time, 
                                    steps_per_epoch=(step_for_training),
                                    epochs=epochs,
                                    validation_data=generator_validation_time,                                 
                                    validation_steps=(step_for_validation),              
                                    callbacks=[checkpointer],
                                    class_weight=class_weights)

# Save the history
with open('/home/jianxig/CNN_LSTM_17/30_timesteps/log_0_aug_w_weights_Jan_23.json', 'w') as file_write:
   json.dump(model_history.history, file_write)
   