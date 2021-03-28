import json
import pandas as pd
from collections import Counter
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

train_data_dir = '/home/jianxig/training_set_17_combo/frames/'
validation_data_dir = '/home/jianxig/validation_set_17_combo/frames/'
csv_file_training = "/home/jianxig/training_set_17_combo/labels_for_training(d_17).csv"
csv_file_validation = "/home/jianxig/validation_set_17_combo/labels_for_validation(d_17).csv"
img_width = 112
img_height = 112 
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
# Note: The preprocessing method is the same as VGG16
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


# (Validation) Load the filenames and labels
dataFrame_valid=pd.read_csv(csv_file_validation, delimiter=',')
dataFrame_valid.Labels=dataFrame_valid.Labels.astype(str) 

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

# Construct the network
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

# Display the architecture
model.summary()

# Save the architecture of the network
with open('/home/jianxig/CNN_17/dataset_17/16_32_32/arch_Jan_22.json','w') as f:
    f.write(model.to_json())
    
# Configure the learning process of the customized network
model.compile(optimizer=Adam(lr = 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Save the optimal weights 
checkpointer = ModelCheckpoint(filepath='/home/jianxig/CNN_17/dataset_17/16_32_32/bestWeights_1_aug_Jan_22.hdf5', 
                               monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')

# Train the customized network (weights are used)
step_for_training = len(generator_train)
step_for_validation = len(generator_validation)
model_history = model.fit_generator(generator_train, 
                                    steps_per_epoch=(step_for_training),
                                    epochs=epochs,
                                    validation_data=generator_validation,                                     
                                    validation_steps=(step_for_validation),              
                                    callbacks=[checkpointer],               
                                    class_weight=class_weights)

# Save the history
with open('/home/jianxig/CNN_17/dataset_17/16_32_32/log_1_aug_Jan_22.json', 'w') as file_write:
   json.dump(model_history.history, file_write)