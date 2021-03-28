import numpy as np
import matplotlib.pyplot as plt    
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from keras.applications.vgg16 import preprocess_input


csv_file_test='/home/jianxig/test_set_17_combo/labels_for_test(d_17).csv'
test_data_dir = "/home/jianxig/test_set_17_combo/frames/"
img_width = 224
img_height = 224
batch_size = 1 # The batch size should be 1, which simplifys the procedure of getting the labels. 
nb_test_samples=6190

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

# Build the VGG16 network. Note that only the convolutional part of the network is initialized. 
model = VGG16(include_top=False, weights=None, input_shape=(224,224,3)) 

# Add layers on the top of VGG 16 
flat_1 = Flatten()(model.layers[-1].output)
dense_1 = Dense(256, activation='relu')(flat_1)
dense_11 = Dropout(0.5)(dense_1)
dense_2 = Dense(256, activation='relu')(dense_1)
dense_22 = Dropout(0.5)(dense_2)
output_1 = Dense(3, activation='softmax')(dense_2)

# Construct the modified model
final_model = Model(inputs = model.inputs, outputs = output_1)

# Show the final model
final_model.summary()

# Load the optimal weights
final_model.load_weights('/home/jianxig/vgg16/w_dropout/bestWeights_1_aug_256_256_3_Jan_21.hdf5')

# Configure the learning process of the neural network
final_model.compile(optimizer=Adam(lr = 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


# Evaluate the network by the test set
final_model.evaluate(generator_test, batch_size=batch_size, verbose=1, steps=len(generator_test))


# Make predictions for the test images
predictions = final_model.predict(generator_test, batch_size=batch_size, verbose=1, steps=len(generator_test))


# Generate the confusion matrix
predicted_result=np.argmax(predictions, axis=1)
true_label=np.argmax(test_labels_df_1, axis=1)
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
number_of_frames=[3903, 1117, 1170]
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




