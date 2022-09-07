# Deep Learning for Needle Detection in a Cannulation Simulator

This is the Keras implementation of the paper *Deep Learning for Needle Detection in a Cannulation Simulator*. If you build projects based on the code, please cite the paper. The bibtex is:

```
@article{gao2021deep,
  title={Deep Learning for Needle Detection in a Cannulation Simulator},
  author={Gao, Jianxin and Lin, Ju and Kil, Irfan and Singapogu, Ravikiran B and Groff, Richard E},
  journal={arXiv e-prints},
  pages={arXiv--2105},
  year={2021}
}
```

## Data sets and network parameters 
Our networks are trained and tested using the [data sets](https://drive.google.com/drive/folders/1m18R03A3EDoURAUM184zxrcUN4C1Eieb?usp=sharing).

During the training process, we recorded the network parameters with respect to the highest validation accuracy. To duplicate our results, please download those [parameters](https://drive.google.com/drive/folders/1D0HNDkNfcTo97wkUPlXHdxkag2i4PnqO?usp=sharing) and load them to the corresponding network model.

## Video demo
To verify the proposed method, we processed a video using CRNN(30ts). This video records an insertion at location FC (i.e., Front Center). The ground truth and the predicted results are shown at the bottom right of the video. 
The network detects needle puncture and infiltration by classifying video frames into 3 classes, namely:
- NoNeedle (shown in green): The needle tip is not in the fistula.
- Fist (shown in blue): The needle tip is in the fistula.
- Infil (shown in red): The needle tip has infiltrated the fistula.

https://user-images.githubusercontent.com/59490151/116702743-7b605a00-a997-11eb-96de-17c8098f3b70.mp4


## System requirements for training and testing the networks
Our networks are trained and tested on a system including:
1. Python 3.6.10
2. OpenCV 3.1.0 
3. TensorFlow 2.2.0
4. Keras 2.4.3
5. CUDA V10.2.89
6. cudnn 8.0.0.180

## System requirements for real-time video processing
Our real-time experiments are accomplished on a system including:
1. Python 3.6.7
2. TensorFlow-gpu 1.8.0 
3. Keras 2.1.5 
4. OpenCV 4.5.1.48 
5. Ffmpeg 4.3.1 
6. Pandas 1.1.3 
8. Pillow 8.0.0 
9. h5py 2.10.0 
10. CUDA V9.0.176
11. cudnn 9.0
