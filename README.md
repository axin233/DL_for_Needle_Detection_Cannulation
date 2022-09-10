# Deep Learning for Needle Detection in a Cannulation Simulator
<!--
This is the Keras implementation of the paper *Deep Learning for Needle Detection in a Cannulation Simulator*. If you build projects based on the code, please cite the paper. The BibTeX is:

```
@article{gao2021deep,
  title={Deep Learning for Needle Detection in a Cannulation Simulator},
  author={Gao, Jianxin and Lin, Ju and Kil, Irfan and Singapogu, Ravikiran B and Groff, Richard E},
  journal={arXiv e-prints},
  pages={arXiv--2105},
  year={2021}
}

```
-->

## Project description
Cannulation is one of the major steps during hemodialysis. To improve clinicians' cannulation skills and avoid putting patients at risk, we design a simulator (shown in Fig. 1a-c) for cannulation training. Our low-cost cannulation simulator uses deep learning techniques for needle puncture and infiltration detection. The detailed description of the simulator can be found at [^2021paper].

![cropped img](https://user-images.githubusercontent.com/59490151/189487389-c5d335cf-efd9-44f3-88aa-ee11499a84dc.PNG)

[^2021paper]: Gao, Jianxin, Ju Lin, Irfan Kil, Ravikiran B. Singapogu, and Richard E. Groff. "Deep Learning for Needle Detection in a Cannulation Simulator." In 2021 International Symposium on Medical Robotics (ISMR), pp. 1-7. IEEE, 2021.

## Video demo
To verify the proposed method, we processed a video using CRNN(30ts)[^2021paper]. This video records an insertion at location FC (i.e., Front Center)[^2021paper]. The ground truth and the predicted results are shown at the bottom right of the video. 

The network detects needle puncture and infiltration by classifying video frames into 3 classes, namely:
- NoNeedle[^2021paper] (shown in green): The needle tip is not in the fistula.
- Fist[^2021paper] (shown in blue): The needle tip is in the fistula.
- Infil[^2021paper] (shown in red): The needle tip has infiltrated the fistula.

https://user-images.githubusercontent.com/59490151/116702743-7b605a00-a997-11eb-96de-17c8098f3b70.mp4

## Data sets and network parameters 
Our networks are trained and tested using the [data sets](https://drive.google.com/drive/folders/1m18R03A3EDoURAUM184zxrcUN4C1Eieb?usp=sharing).

During the training process, we recorded the network parameters for the highest validation accuracy. To duplicate our results, please download those [parameters](https://drive.google.com/drive/folders/1D0HNDkNfcTo97wkUPlXHdxkag2i4PnqO?usp=sharing) and load them to the corresponding network model.

## System requirements for training and testing the networks
Our networks are trained and tested on a system with:
1. Keras 2.4.3  
2. TensorFlow 2.2.0
3. OpenCV 3.1.0
4. CUDA V10.2.89

## System requirements for real-time video processing
Our real-time experiments are accomplished on a system with:
1. Keras 2.1.5 
2. TensorFlow-GPU 1.8.0 
3. OpenCV 4.5.1.48
4. CUDA V9.0.176
