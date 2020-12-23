import numpy as np
from numpy.core.arrayprint import dtype_short_repr
from numpy.core.fromnumeric import reshape
import tensorflow as tf
from tensorflow.python.ops.functional_ops import Gradient
from config import *


def create_label(label: np.ndarray, annotation: str) -> np.ndarray:
    elements = annotation.split(" ")
    grid_x = int(elements[0])
    grid_y = int(elements[1])
    x = float(elements[2])
    y = float(elements[3])
    w = float(elements[4])
    h = float(elements[5])
    class_vector = [0 for i in range(len(CLASS_LIST))]
    class_vector[int(elements[6])] = 1
    label[grid_x, grid_y, 0] = 1
    label[grid_x, grid_y, 1] = x
    label[grid_x, grid_y, 2] = y
    label[grid_x, grid_y, 3] = w
    label[grid_x, grid_y, 4] = h
    label[grid_x, grid_y, 5:] = class_vector
    return label

def iou_calc(target_box: tf.Tensor, pred_boxes: tf.Tensor) -> tf.Tensor:
    '''
        # IOU Calculation
        ---
        ## Params
        ### target_box
        tf.Tensor with shape of (None, GRID_COUNT, GRID_COUNT, BOX_PER_GRID, 4)\n
        ### pred_boxes
        tf.Tensor with shape of (None, GRID_COUNT, GRID_COUNT, BOX_PER_GRID, 4)

        ## Return\n
        tf.Tensor with shape of (None, GRID_COUNT, GRID_COUNT, BOX_PER_GRID)
    '''
    batch_size = target_box.shape[0]
    off_set = tf.range(GRID_COUNT, dtype=tf.float32)
    off_set = tf.tile(tf.reshape(
                        off_set, 
                        (1, 1, GRID_COUNT, 1)),
                      (batch_size, GRID_COUNT, 1, BOX_PER_GRID))
    
    target_left_side = (target_box[...,0] + off_set) / GRID_COUNT - target_box[...,2] / 2
    target_right_side = (target_box[...,0] + off_set) / GRID_COUNT + target_box[...,2] / 2
    target_upper_side = (target_box[...,1] + off_set) / GRID_COUNT - target_box[...,3] / 2
    target_lower_side = (target_box[...,1] + off_set) / GRID_COUNT + target_box[...,3] / 2

    pred_left_side = (pred_boxes[...,0] + off_set) / GRID_COUNT - pred_boxes[...,2] / 2
    pred_right_side = (pred_boxes[...,0] + off_set) / GRID_COUNT + pred_boxes[...,2] / 2
    pred_upper_side = (pred_boxes[...,1] + off_set) / GRID_COUNT - pred_boxes[...,3] / 2
    pred_lower_side = (pred_boxes[...,1] + off_set) / GRID_COUNT + pred_boxes[...,3] / 2

    inter_height = tf.minimum(tf.maximum(target_lower_side - pred_upper_side, 0), 
                              tf.minimum(tf.maximum(pred_lower_side - target_upper_side, 0),
                              tf.minimum(pred_lower_side - pred_upper_side,
                              target_lower_side - target_upper_side)))
    inter_width = tf.minimum(tf.maximum(target_right_side - pred_left_side, 0),
                             tf.minimum(tf.maximum(pred_right_side - target_left_side, 0),
                             tf.minimum(pred_right_side - pred_left_side,
                             target_right_side - target_left_side)))
    inter_area = inter_height * inter_width
    total_area = pred_boxes[...,2] * pred_boxes[...,3] + target_box[...,2] * target_box[...,3] - inter_area

    return inter_area / total_area

def parse_prediction(preds: tf.Tensor) -> list:
    '''
        YoloNet Output Parser
        ---
        ## Form :
        [<-- Classification -->|<-- Boxes Confidence -->|<-- Boxes Location -->]\n
        ### Classification : 
        >>> class_score[..., :] # One-hot vector for classification
        
        ### Boxes Confidence :
        >>> confidence[..., 0] # Confidence for first box
        >>> confidence[..., 1] # Confidence for second box
        
        ### Boxes Location :
        >>> boxes[...,box_num ,0:4] # Box location 1 (x, y, w, h)
    '''
    batch_cnt = preds.shape[0]
    class_splitter = GRID_COUNT * GRID_COUNT * CLASS_COUNT
    class_score = tf.reshape(preds[...,:class_splitter], 
                                (batch_cnt, GRID_COUNT, GRID_COUNT, CLASS_COUNT))
    conf_splitter = class_splitter + GRID_COUNT * GRID_COUNT * BOX_PER_GRID
    confidence = tf.reshape(preds[...,class_splitter:conf_splitter], 
                            (batch_cnt, GRID_COUNT, GRID_COUNT, BOX_PER_GRID))
    boxes = tf.reshape(preds[...,conf_splitter:], 
                       (batch_cnt, GRID_COUNT, GRID_COUNT, BOX_PER_GRID, 4))
    return class_score, confidence, boxes

def parse_label(labels: tf.Tensor) -> list:
    '''
    Label Parser
    ---
    ### Class note : 
    >>> classification[..., :] # One-hot vector for classification

    ### Object Existence : 
    >>> hasObj[...,:] # 1.0 for true, 0.0 for false

    ### Box Location :
    >>> box[...,:] # Box location 1 (x, y, w, h)
    '''
    batch_cnt = labels.shape[0]
    has_obj = labels[..., 0]
    box = labels[..., 1:5]
    box = tf.reshape(box, (batch_cnt, GRID_COUNT, GRID_COUNT, 1, 4))
    box = tf.tile(box, [1, 1, 1, BOX_PER_GRID, 1])
    classification = labels[..., 5:]
    return classification, has_obj, box
