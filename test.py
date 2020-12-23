from config import *
import utils
import os
import numpy as np
import cv2
import tensorflow as tf 

def read_train_data():
    file_list = os.listdir('./data/train')
    labels = []
    imgs = []
    for file in file_list:
        label_file_path = './data/train/' + file
        with open(label_file_path) as f:
            lines = f.readlines()
        label = np.zeros((GRID_COUNT, GRID_COUNT, (CLASS_COUNT + 5)))
        for line in lines:
            label = utils.create_label(label, line)
        labels.append(label)

        img_file_path = TRAIN_IMG_FOLDER + '/' + file.replace(".txt", '.jpg')
        img = cv2.imread(img_file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return labels, imgs

def gen_training():
    file_list = os.listdir('./data/train')
    labels_path_list = []
    imgs_path_list = []
    for file in file_list:
        labels_path_list.append('./data/train/' + file)
        imgs_path_list.append(TRAIN_IMG_FOLDER + '/' + file.replace(".txt", '.jpg'))
    index = 0
    while  True:
        image = cv2.imread(imgs_path_list[index])
        image = cv2.resize(image, (224, 224))
        with open(labels_path_list[index]) as f:
            lines = f.readlines()
        label = np.zeros((GRID_COUNT, GRID_COUNT, (5 + CLASS_COUNT)))
        for line in lines:
            elements = line.split(" ")
            grid_x = int(elements[0])
            grid_y = int(elements[1])
            label[grid_x, grid_y, 0] = 1
            label[grid_x, grid_y, 1] = float(elements[2])
            label[grid_x, grid_y, 2] = float(elements[3])
            label[grid_x, grid_y, 3] = float(elements[4])
            label[grid_x, grid_y, 4] = float(elements[5])
            class_vector = [0 for i in range(len(CLASS_LIST))]
            class_vector[int(elements[6])] = 1
            label[grid_x, grid_y, 5:] = class_vector
        yield  (image, label)
        index += 1
        if index == len(labels_path_list):
            break
            # index = 0
