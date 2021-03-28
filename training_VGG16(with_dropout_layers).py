
import json
import pandas as pd
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import preprocess_input

# For batch_size=6
train_data_dir = '/home/jianxig/training_set_17_combo/frames/'
validation_data_dir = '/home/jianxig/validation_set_17_combo/frames/'
csv_file_training = "/home/jianxig/training_set_17_combo/labels_for_training(d_17).csv"
csv_file_validation = "/home/jianxig/validation_set_17_combo/labels_for_validation(d_17).csv"
img_width = 224
img_height = 224
batch_size = 6
nb_validation_samples=2910
nb_train_samples=49962  
epochs=50

# Load the filenames and labels
dataFrame=pd.read_csv(csv_file_training, delimiter=',')
dataFrame.Labels=dataFrame.Labels.astype(str) # Change the datatype of the column 'Labels'

# Specify the class names
class_names=['Preparation','In fistula','infiltration']

# Apply data augmentation to the training set. 
datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, 
        rotation_range=10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.5, 1.5])


# Preprocessing the training set.
generator_train = datagen.flow_from_dataframe(
        dataFrame,         
        directory = train_data_dir,        
        x_col = 'File_Name',       
        y_col = 'Labels',       
        target_size = (img_height, img_width),       
        class_mode='categorical',       
        batch_size=batch_size,       
        shuffle=True) 

# Calculate weights for unbalanced classes within training set
counter = Counter(dataFrame.iloc[:, 1].tolist()) # Count how many images within each class
max_val = float(max(counter.values()))
class_weights = {int(class_id,10) : max_val/num_images for class_id, num_images in counter.items()}

# Load the filenames and labels
dataFrame_valid=pd.read_csv(csv_file_validation, delimiter=',')
dataFrame_valid.Labels=dataFrame_valid.Labels.astype(str) # Change the datatype of the column 'Labels'

# Preprocess the validation set, but avoid data augmentation
datagen_val = ImageDataGenerator(preprocessing_function=preprocess_input)

# Preprocessing the validation set.
generator_validation = datagen_val.flow_from_dataframe(
        dataFrame_valid,       
        directory = validation_data_dir,       
        x_col = 'File_Name', 
        y_col = 'Labels',
        target_size = (img_height, img_width),        
        class_mode='categorical',        
        batch_size=batch_size,        
        shuffle=True)  

# Build the VGG16 network. Note that only the convolutional part of the network is instantiated. 
model = applications.VGG16(include_top=False, weights='imagenet', input_shape=(224,224,3)) 

# mark loaded layers as not trainable
for layer in model.layers:
	layer.trainable = False 

# Add layers on the top of VGG 16 
flat_1 = Flatten()(model.layers[-1].output)
dense_1 = Dense(256, activation='relu')(flat_1)
dense_11 = Dropout(0.5)(dense_1)
dense_2 = Dense(256, activation='relu')(dense_11)
dense_22 = Dropout(0.5)(dense_2)
output_1 = Dense(3, activation='softmax')(dense_22)

# Construct the modified model
final_model = Model(inputs = model.inputs, outputs = output_1)

# Show the final model
final_model.summary()


# Save the architecture of the customized network
with open('/home/jianxig/vgg16/w_dropout/arch_256_256_3_Jan_21.json','w') as f:
    f.write(final_model.to_json())


# Configure the learning process of the customized network
final_model.compile(optimizer=Adam(lr = 0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

# Save the optimal weights 
checkpointer = ModelCheckpoint(filepath='/home/jianxig/vgg16/w_dropout/bestWeights_1_aug_256_256_3_Jan_21.hdf5', 
                               monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')


# Train the customized network (weights are used)
step_for_training = len(generator_train)
step_for_validation = len(generator_validation)
model_history = final_model.fit_generator(generator_train, 
                                          steps_per_epoch=(step_for_training),
                                          epochs=epochs,
                                          validation_data=generator_validation, 
                                          validation_steps=(step_for_validation),
                                          callbacks=[checkpointer], 
                                          class_weight=class_weights)

# Save the history
with open('/home/jianxig/vgg16/w_dropout/log_1_aug_256_256_3_Jan_21.json', 'w') as file_write:
   # write json data into file
   json.dump(model_history.history, file_write)

# List all data in history
print(model_history.history.keys())
