# %%
import tensorflow as tf

print(tf.__version__)
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from time import sleep

# %%
targets_dict = {
    0: "Thumb Down",
    1: "Stop Sign",
    2: "Sliding Two Fingers Left",
    3: "Sliding Two Fingers Right",
    4: "No gesture",
    5: "Thumb Up"
}


# %%
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def normaliz_data2(v):
    normalized_v = v / np.sqrt(np.sum(v ** 2))
    return normalized_v

# %%
def normaliz_data(np_data):
    # Normalisation
    scaler = StandardScaler()
    #scaled_images  = normaliz_data2(np_data)
    scaled_images = np_data.reshape(-1, 30, 64, 64, 1)
    return scaled_images


# %%


class Conv3DModel(tf.keras.Model):
    def __init__(self):
        super(Conv3DModel, self).__init__()
        # Convolutions
        self.conv1 = tf.compat.v2.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', name="conv1",
                                                      data_format='channels_last')
        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')
        self.conv2 = tf.compat.v2.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv1",
                                                      data_format='channels_last')
        self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), data_format='channels_last')

        # LSTM & Flatten
        self.convLSTM = tf.keras.layers.ConvLSTM2D(40, (3, 3))
        self.flatten = tf.keras.layers.Flatten(name="flatten")

        # Dense layers
        self.d1 = tf.keras.layers.Dense(128, activation='relu', name="d1")
        self.out = tf.keras.layers.Dense(6, activation='softmax', name="output")

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.convLSTM(x)
        # x = self.pool2(x)
        # x = self.conv3(x)
        # x = self.pool3(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.out(x)


# %%
class Prediction:
    def __init__(self):
        new_model = Conv3DModel()
        # %%
        new_model.compile(loss='sparse_categorical_crossentropy',
                          optimizer=tf.keras.optimizers.RMSprop())

        # %%
        new_model.load_weights('tf_model/3d-cnn-basic')
        print('loaded')
        self.model = new_model

    def predict(self, frames):
        frame_to_predict = np.array(frames, dtype=np.float32)
        frame_to_predict = normaliz_data(frame_to_predict)
        # print(frame_to_predict)
        predict_val = self.model.predict(frame_to_predict)
        class_num = np.argmax(predict_val)
        #print(class_num)
        return class_num