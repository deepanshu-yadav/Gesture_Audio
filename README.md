# Change songs through hand gestures
This project lets you control your playlist, which is playing at your laptop
without getting up from bed.  

There are five types of gestures / symbols supported
1. Play
2. Pause
3. Stop
4. Next 
5. Previous

# Platform supported
Currently Ubuntu 18.04 is only supported.
In future there are plans to extend support to other platforms.

# Installation
1. Run the **requirements.sh** using `./requirements.sh`
2. Install open vino from [here](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html).

# Running
-  Open VINO installtion is usually found **/opt/intel**.
So type the following command 
```
source /opt/intel/openvino/bin/setupvars.sh
```
-  Now in this repository go to Lazy_Change directory and type
```
python3 Lazy_Change.py
```



# Notebooks
1. For data [preparation](notebooks/jester_data_preparation.ipynb) and preprocessing. 
2. For training the model the model at [colab](notebooks/train_jester_on_colab.ipynb).
3. For testing the model after training on a [webcam](notebooks/test_tensorflow_model.ipynb).


# Credits 
A Big Thank You to these guys 
1. Modified  [his](https://github.com/chaNcharge/PyTunes/)  code for GUI.
2. Took help from this [post](https://iosoft.blog/cam-display/) for displaying video feed in PyQt5.
3. Took songs from [him](https://github.com/yashshah2609/Emotion-Based-music-player).
4. Converted keras model to a freezed tensorflow model from [here](https://software.intel.com/en-us/forums/intel-distribution-of-openvino-toolkit/topic/820599).
5. Thank you [Udacity](https://udacity.com) for making such wonderful tutorials.
6. The best tutorial on Intel NCS is [here](https://www.pyimagesearch.com/2019/04/08/openvino-opencv-and-movidius-ncs-on-the-raspberry-pi/) .
7. The dataset is available [here](https://20bn.com/datasets/jester/v1).
8. Modified his [code](https://github.com/anasmorahhib/3D-CNN-Gesture-recognition/blob/master/main.ipynb) to improve the accuracy of of my earlier model. Thanks a ton to him.

# Future Work
1. Better accurate model and more complex gestures(May be a combination of gestures).
2. Support for NCS 2 for faster inference.
3. Support for Raspberry Pi.
