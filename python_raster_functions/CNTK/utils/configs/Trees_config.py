from easydict import EasyDict as edict

__C = edict()
__C.DATA = edict()
cfg = __C

__C.DATA.DATASET = "Trees"
__C.DATA.MAP_FILE_PATH = "../../DataSets/Trees"
__C.DATA.CLASS_MAP_FILE = "class_map.txt"
__C.DATA.TRAIN_MAP_FILE = "train_img_file.txt"
__C.DATA.TEST_MAP_FILE = "test_img_file.txt"
__C.DATA.TRAIN_ROI_FILE = "train_roi_file.txt"
__C.DATA.TEST_ROI_FILE = "test_roi_file.txt"
__C.DATA.NUM_TRAIN_IMAGES = 31
__C.DATA.NUM_TEST_IMAGES = 1
__C.DATA.PROPOSAL_LAYER_SCALES = [8, 16, 32]