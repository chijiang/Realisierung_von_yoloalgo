BOX_PER_GRID = 2
GRID_COUNT = 7
IMG_SIZE = 448
CLASS_LIST = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
              'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
              'dog', 'horse', 'motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor']
CLASS_COUNT = len(CLASS_LIST)

# directories
TRAIN_DIR = "../pascal_VOC/VOCdevkit_trainval/VOC2012"
TEST_DIR = "../pascal_VOC/VOCdevkit_test/VOC2012"

TRAIN_IMG_FOLDER = TRAIN_DIR + "/JPEGImages"
TEST_IMG_FOLDER = TEST_DIR + "/JPEGImages"

TRAIN_ANNOTATIONS_FOLDER = TRAIN_DIR + "/Annotations"
TEST_ANNOTATIONS_FOLDER = TEST_DIR + "/Annotations"

TRAIN_SET_FOLDER = TRAIN_DIR + "/ImageSets/Main"
TEST_SET_FOLDER = TEST_DIR + "/ImageSets/Main"

# Training params
BATCH_SIZE = 16