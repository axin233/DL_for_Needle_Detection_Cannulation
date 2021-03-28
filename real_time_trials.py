import numpy as np
import cv2
from PIL import Image 
import subprocess as sp  
from time import perf_counter
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, TimeDistributed,LSTM
 
timesteps_for_LSTM = 30

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
cnn_model.load_weights('E:/Fall 2020/PhD research/neural network/paper_updated/CNN_17/16_32_32/bestWeights_1_aug_Jan_22.hdf5')

# Replace the output layer of the cnn with a 32-neuron dense layer
dense_1=Dense(32, activation='relu')(cnn_model.layers[-2].output)
modified_cnn=Model(inputs = cnn_model.inputs, outputs=dense_1)

# Freeze the weights within the cnn layers
layer=0
for layer in modified_cnn.layers[:-1]:
    layer.trainable = False
    
# Display the modified cnn    
modified_cnn.summary()

# Define the CRNN
model=Sequential()

# Add the cnn model
model.add(TimeDistributed(modified_cnn, input_shape=(timesteps_for_LSTM,112,112,3)))

# The LSTM layer
model.add(LSTM(32, dropout=0.5))

# The output layer
model.add(Dense(3, activation='softmax'))

# Display the architecture of CRNN
model.summary()

# Load the optimal weights
model.load_weights('E:/Fall 2020/PhD research/neural network/paper_updated/CNN_LSTM_17/30_ts/bestWeights_0_aug_w_weights_Jan_23.hdf5')

# Configure the learning process of the neural network
model.compile(optimizer=Adam(lr = 0.0001),
                  loss='categorical_crossentropy', metrics=['accuracy'])

# Construct a duplicated cnn model
cnn_m_1=Sequential()

# The first CNN block
cnn_m_1.add(
    Conv2D(input_shape=(112,112,3), filters=16, kernel_size=(3,3), strides=(1, 1), padding="same", activation="relu"))
cnn_m_1.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# The seccond CNN block
cnn_m_1.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding="same", activation="relu"))
cnn_m_1.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# The third CNN block
cnn_m_1.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding="same", activation="relu"))
cnn_m_1.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

# Convert the 2D features to the 1D features
cnn_m_1.add(Flatten())

# The output layer
cnn_m_1.add(Dense(3, activation='softmax'))

# Load the pre-trained weights of the cnn model
cnn_m_1.load_weights('E:/Fall 2020/PhD research/neural network/paper_updated/CNN_17/16_32_32/bestWeights_1_aug_Jan_22.hdf5')

# Replace the output layer of cnn to a 32-neuron dense layer
dense_m_1=Dense(32, activation='relu')(cnn_m_1.layers[-2].output)

# Define the modified cnn
modified_cnn_1=Model(inputs = cnn_m_1.inputs, outputs=dense_m_1)

# Freeze the weights within the cnn layers
layer=0
for layer in modified_cnn_1.layers[:-1]:
    layer.trainable = False
    
# Display the modified cnn    
modified_cnn_1.summary()

# Define the first half of CRNN for real-time processing
final_cnn_model=Sequential()

# Add the cnn model
final_cnn_model.add(TimeDistributed(modified_cnn_1, input_shape=(1,112,112,3)))

# Load the weights 
final_cnn_model.layers[0].set_weights(model.layers[0].get_weights())

# Define the second half of CRNN for real-time processing
final_model_lstm=Sequential()
# The LSTM layer.
final_model_lstm.add(LSTM(32, dropout=0.5, input_shape=(30,32)))

# The output layer
final_model_lstm.add(Dense(3, activation='softmax'))

# Display the architecture of the second half of CRNN
final_model_lstm.summary()

# Load the weights 
final_model_lstm.layers[0].set_weights(model.layers[1].get_weights())
final_model_lstm.layers[1].set_weights(model.layers[2].get_weights())

# For real-time processing
img_features_seq=[]
real_time_result=[]
test_seq=[]
real_time_result_single=10
file_path='F:/Data_for_Research/Irfan/dataset(Jan_19)/'

# Experiment index
trial_ind='36'

# To record frame rates
frameNum=1
# To check the frame rate
frame_rate_n=file_path + 'results(Jan_24)/' + 'frameRate_' + trial_ind + '.txt'
frame_rate_f=open(frame_rate_n, 'w')
t_start=perf_counter()

# Turn on a camera
cap = cv2.VideoCapture(1)
if cap.isOpened()==False:
    print("Error! Cannot turn on the camera.")

# For ffmpeg library
output_video=file_path + 'internal_videos/' + 'v_' + trial_ind + '.avi'
ffmpeg_bin = 'ffmpeg.exe' # The name (or path) of the ffmpeg library
command = [ffmpeg_bin,
        '-y', # (optional) overwrite output file if it exists
        '-f', 'rawvideo', #The format
        '-vcodec','rawvideo',
        '-s', '640x480', # The size of one frame
        '-pix_fmt', 'bgr24', # Remember OpenCV uses bgr format
        '-r', '30', # Frame rate (frames per second)
        '-i', '-', # The imput comes from a pipe
        '-an', # Tells FFMPEG not to expect any audio
        '-vcodec', 'huffyuv',
        '-pix_fmt', 'rgb24',
        '-loglevel', 'quiet',
        output_video ]
pipe = sp.Popen(command, stdin=sp.PIPE, stdout=None, stderr=None)

# If the camera has been turned on 
while(cap.isOpened()):
    
    # Capture frame-by-frame. Note that the order of channels is BGR
    ret, frame = cap.read()
    
    # If frames can be obtained from the camera
    if ret==True:
        
        # Convert a numpy array to PIL image
        img=Image.fromarray(frame)
            
        # Resize the image by PIL library
        scaled_img = img.resize((112,112), Image.NEAREST)
            
        # Change the data type of image
        float_img=np.float32(scaled_img)
            
        # Zero-center each color channel with respect to 
        # the ImageNet dataset (BGR [103.939, 116.779, 123.68])
        normalized_img=np.zeros((112, 112, 3), dtype='float32')
        normalized_img[:,:,0]=float_img[:,:,0] - 103.939
        normalized_img[:,:,1]=float_img[:,:,1] - 116.779
        normalized_img[:,:,2]=float_img[:,:,2] - 123.68
            
        # Change the matrix from (112, 112, 3) to (1, 112, 112 ,3)
        reshaped_input_cnn_temp = np.expand_dims(normalized_img, axis=0)
        # Change the matrix from (1, 112, 112, 3) to (1, 1, 112, 112 ,3)
        reshaped_input_cnn = np.expand_dims(reshaped_input_cnn_temp, axis=0)
            
        # Extract features by the first half of CRNN
        img_features_1 = final_cnn_model.predict(reshaped_input_cnn,batch_size=1,verbose=0)
            
        # Remove the first dimension
        img_features_2 = np.squeeze(img_features_1, axis=0)
   
        # Remove the second dimension
        img_features = np.squeeze(img_features_2, axis=0)
        
        # Construct the time series data
        if frameNum<=timesteps_for_LSTM:
            img_features_seq.append(img_features)
        else:
            img_features_seq.pop(0)
            img_features_seq.append(img_features)
            
        # Generate prediction results using the second half of CRNN
        if frameNum>=timesteps_for_LSTM:
            # Convert a list to a numpy array
            np_img_features_seq=np.array(img_features_seq)
            
            # Convert an array from (timesteps_for_LSTM, 32) to (1, timesteps_for_LSTM, 32)
            reshape_input_lstm = np.expand_dims(np_img_features_seq, axis=0)
            
            # Generate the results
            real_time_predict = final_model_lstm.predict(reshape_input_lstm,batch_size=1,verbose=0)
            real_time_result_single = np.argmax(real_time_predict)
            real_time_result.append(real_time_result_single)
        
        # Check the frame rate
        t_end=perf_counter() # record the current time
        t_diff = t_end - t_start
        frame_rate_f.write('{}{}{}{}{}\n'.format(frameNum,', ',t_diff,', ',real_time_result_single))
        
        # save the frame by ffmpeg library
        frame_str=frame.tobytes()
        pipe.stdin.write(frame_str)
        
        # Increase the frame number
        frameNum=frameNum+1
        
        # Display frames
        cv2.imshow('frame',frame)
        
        # press character 's' to stop recording a video
        if cv2.waitKey(1) == 115: 
            pipe.stdin.close()
            pipe.wait()
            break
    else:
        break

# When everything is done, release cv2.VideoCapture
cap.release()
cv2.destroyAllWindows()  
frame_rate_f.close()
