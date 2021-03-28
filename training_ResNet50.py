import json
import pandas as pd
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import ResNet50

# For batch_size=6
train_data_dir = '/home/jianxig/training_set_17_combo/frames/'
validation_data_dir = '/home/jianxig/validation_set_17_combo/frames/'
csv_file_training = "/home/jianxig/training_set_17_combo/labels_for_training(d_17).csv"
csv_file_validation = "/home/jianxig/validation_set_17_combo/labels_for_validation(d_17).csv"
img_width = 224
img_height = 224
batch_size_1 = 6
nb_validation_samples=2910 
nb_train_samples=49962 
epochs=50

# Load the filenames and labels
dataFrame=pd.read_csv(csv_file_training, delimiter=',')
dataFrame.Labels=dataFrame.Labels.astype(str) 

# Specify the class names
class_names=['Preparation','In fistula','infiltration']

# Preprocess the training set (With data augmentation)
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
        batch_size=batch_size_1,       
        shuffle=True) 

# Calculate weights for unbalanced classes within training set
counter = Counter(dataFrame.iloc[:, 1].tolist()) # Count how many images within each class
max_val = float(max(counter.values()))
class_weights = {int(class_id,10) : max_val/num_images for class_id, num_images in counter.items()}

# (Validation) Load the filenames and labels
dataFrame_valid=pd.read_csv(csv_file_validation, delimiter=',')
dataFrame_valid.Labels=dataFrame_valid.Labels.astype(str) 

# (Validation) Preprocess the validation set. (Without data augmentation)
datagen_val = ImageDataGenerator(preprocessing_function=preprocess_input)
    

# (Validation) Preprocessing the validation set.
generator_validation = datagen_val.flow_from_dataframe(
        dataFrame_valid,         
        directory = validation_data_dir,        
        x_col = 'File_Name',         
        y_col = 'Labels',        
        target_size = (img_height, img_width),        
        class_mode='categorical',        
        batch_size=batch_size_1,        
        shuffle=True)  


# Build the ResNet. Note that only the convolutional part is used
# The output layers will be created later.
ResNet_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3)) 

# Freeze the layers
for layer in ResNet_model.layers:
    layer.trainable = False
    

# Add a global average pooling layer
GAP = GlobalAveragePooling2D()(ResNet_model.layers[-1].output)

# Add the output layer
outp = Dense(3, activation='softmax')(GAP)

# Define the customized network
final_model = Model(inputs = ResNet_model.inputs, outputs = outp)

# Display the customized network
final_model.summary()

# Save the architecture of the customized network
with open('/home/jianxig/ResNet50/arch_GAP_O(Jan_21).json','w') as f:
    f.write(final_model.to_json())


# Configure the learning process of the customized network
final_model.compile(optimizer=Adam(lr = 0.0001),loss='categorical_crossentropy', metrics=['accuracy'])

# Save the optimal weights 
checkpointer = ModelCheckpoint(filepath='/home/jianxig/ResNet50/bestWeights_1_aug_GAP_O(Jan_21).hdf5', 
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
with open('/home/jianxig/ResNet50/log_1_aug_GAP_O(Jan_21).json', 'w') as file_write:
   # write json data into file
   json.dump(model_history.history, file_write)