from utils import iou_calc, parse_label, parse_prediction
from config import *
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
from tensorflow.keras import Model

class YoloModel(Model):
    def __init__(self):
        self.in_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        # Conv Layers 0 -> (112, 112, 192)
        self.hidden0 = Conv2D(64, (7, 7), strides=2, activation=tf.nn.leaky_relu)(self.in_layer)
        self.mp0 = MaxPool2D(strides=2)(self.hidden0)
        
        # Conv Layers 1 -> (112, 112, 192)
        self.hidden1 = Conv2D(192, (3, 3), activation=tf.nn.leaky_relu, padding="same")(self.mp0)
        self.mp1 = MaxPool2D(strides=2)(self.hidden1)

        # Conv Layers 2 -> (56, 56, 256)
        self.hidden2 = Conv2D(128, (1, 1), activation=tf.nn.leaky_relu, padding="same")(self.mp1)
        self.hidden2 = Conv2D(256, (3, 3), activation=tf.nn.leaky_relu, padding="same")(self.hidden2)
        self.hidden2 = Conv2D(256, (1, 1), activation=tf.nn.leaky_relu, padding="same")(self.hidden2)
        self.hidden2 = Conv2D(512, (3, 3), activation=tf.nn.leaky_relu, padding="same")(self.hidden2)
        self.mp2 = MaxPool2D(strides=2)(self.hidden2)
        
        # Conv Layers 3 -> (14, 14, 1024)
        self.hidden3 = Conv2D(256, (1, 1), activation=tf.nn.leaky_relu, padding="same")(self.mp2)
        self.hidden3 = Conv2D(512, (3, 3), activation=tf.nn.leaky_relu, padding="same")(self.hidden3)
        self.hidden3 = Conv2D(256, (1, 1), activation=tf.nn.leaky_relu, padding="same")(self.hidden3)
        self.hidden3 = Conv2D(512, (3, 3), activation=tf.nn.leaky_relu, padding="same")(self.hidden3)
        self.hidden3 = Conv2D(256, (1, 1), activation=tf.nn.leaky_relu, padding="same")(self.hidden3)
        self.hidden3 = Conv2D(512, (3, 3), activation=tf.nn.leaky_relu, padding="same")(self.hidden3)
        self.hidden3 = Conv2D(256, (1, 1), activation=tf.nn.leaky_relu, padding="same")(self.hidden3)
        self.hidden3 = Conv2D(512, (3, 3), activation=tf.nn.leaky_relu, padding="same")(self.hidden3)
        self.hidden3 = Conv2D(512, (1, 1), activation=tf.nn.leaky_relu, padding="same")(self.hidden3)
        self.hidden3 = Conv2D(1024, (3, 3), activation=tf.nn.leaky_relu, padding="same")(self.hidden3)
        self.mp3 = MaxPool2D(strides=2)(self.hidden3)

        # Conv Layers 4 -> (7, 7, 1024)
        self.hidden4 = Conv2D(512, (1, 1), activation=tf.nn.leaky_relu, padding="same")(self.mp3)
        self.hidden4 = Conv2D(1024, (3, 3), activation=tf.nn.leaky_relu, padding="same")(self.hidden4)
        self.hidden4 = Conv2D(512, (1, 1), activation=tf.nn.leaky_relu, padding="same")(self.hidden4)
        self.hidden4 = Conv2D(1024, (3, 3), activation=tf.nn.leaky_relu, padding="same")(self.hidden4)
        self.hidden4 = Conv2D(1024, (3, 3), activation=tf.nn.leaky_relu, padding="same")(self.hidden4)
        self.hidden4 = Conv2D(1024, (3, 3), strides=2, activation=tf.nn.leaky_relu)(self.hidden4)

        # Conv Layers 5 -> (7, 7, 1024)
        self.hidden5 = Conv2D(1024, (3, 3), activation=tf.nn.leaky_relu, padding="same")(self.hidden4)
        self.hidden5 = Conv2D(1024, (3, 3), activation=tf.nn.leaky_relu, padding="same")(self.hidden5)

        # FC Layers -> (4096,)
        self.hidden6 = Flatten()(self.hidden5)
        self.hidden6 = Dense(4096, activation=tf.nn.sigmoid)(self.hidden6)

        # Output Layer -> (7,7,30)
        self.out_layer = Dense(GRID_COUNT * GRID_COUNT * (5 * BOX_PER_GRID + CLASS_COUNT), activation=tf.nn.sigmoid)(self.hidden6)

        super(YoloModel, self).__init__(inputs=self.in_layer, outputs=self.out_layer)

    def yolo_loss_fn(self, prediction, label):
        pred_class, confidence, pred_boxes = parse_prediction(prediction)
        true_class, has_obj, true_box = parse_label(label)

        iou = iou_calc(true_box, pred_boxes)
