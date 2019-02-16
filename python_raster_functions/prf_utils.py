'''
Copyright 2018 Esri

Licensed under the Apache License, Version 2.0 (the "License");

you may not use this file except in compliance with the License.

You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software

distributed under the License is distributed on an "AS IS" BASIS,

WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and

limitations under the License.​
'''

import math
import numpy as np


def calculate_rectangle_size_from_batch_size(batch_size):
    """
    Calculate the the number of tiles vertically and horizontally give a batch size
    :param batch_size: the batch size
    :return: number of tiles vertically and horizontally, batch_height and batch_width
    """
    batch_height = int(math.sqrt(batch_size) + 0.5)
    batch_width = int(batch_size / batch_height)

    if batch_height * batch_width > batch_size:
        if batch_height >= batch_width:
            batch_height = batch_height - 1
        else:
            batch_width = batch_width - 1

    if (batch_height + 1) * batch_width <= batch_size:
        batch_height = batch_height + 1
    if (batch_width + 1) * batch_height <= batch_size:
        batch_width = batch_width + 1

    # swap col and row to make a horizontal rect
    if batch_height > batch_width:
        batch_height, batch_width = batch_width, batch_height

    if batch_height * batch_width != batch_size:
        return batch_size, 1

    return batch_height, batch_width


def get_tile_size(model_height, model_width, padding, batch_height, batch_width):
    """
    Calculate the super tile dimension given model height, padding, the number of tiles vertically and horizontally
    :param model_height: the model's height
    :param model_width: the model's width
    :param padding: padding
    :param batch_height: the number of tiles vertically
    :param batch_width: the number of tiles horizontally
    :return: super tile height and width, tile_height and tile_width
    """
    tile_height = (model_height - 2 * padding) * batch_height
    tile_width = (model_width - 2 * padding) * batch_width

    return tile_height, tile_width


def tile_to_batch(pixel_block, model_height, model_width, padding, fixed_tile_size=True, **kwargs):
    """
    Convert a super tile to a batch of mini tiles
    :param pixel_block: the super tile numpy array
    :param model_height: the model's height
    :param model_width: the model's width
    :param padding: padding, the padding in each mini tile
    :param fixed_tile_size: if True, the pixel block is filled to have batch_height x batch_width, batch_height and
                            batch_width are given in kwargs; if False, the pixel block is not filled to have exactly
                            batch_height x batch_width
    :param kwargs: needed to give batch_height and batch_width if fixed_tile_size is True
    :return: batch: the batch numpy array [B, D, H, W]
             batch_height: the number of tiles vertically, if fixed_tile_size is True, this is equal to batch_height in
                           kwargs
             batch_width: the number of tiles horizontally, if fixed_tile_size is True, this is equal to batch_width in
                          kwargs
    """
    inner_width = model_width - 2 * padding
    inner_height = model_height - 2 * padding

    band_count, pb_height, pb_width = pixel_block.shape
    pixel_type = pixel_block.dtype

    if fixed_tile_size is True:
        batch_height = kwargs['batch_height']
        batch_width = kwargs['batch_width']
    else:
        batch_height = math.ceil((pb_height - 2 * padding) / inner_height)
        batch_width = math.ceil((pb_width - 2 * padding) / inner_width)

    batch = np.zeros(shape=(batch_width * batch_height, band_count, model_height, model_width), dtype=pixel_type)
    for b in range(batch_width * batch_height):
        y = int(b / batch_width)
        x = int(b % batch_width)

        # pixel block might not be the shape (band_count, model_height, model_width)
        sub_pixel_block = pixel_block[:, y * inner_height: y * inner_height + model_height,
                    x * inner_width: x * inner_width + model_width]
        sub_pixel_block_shape = sub_pixel_block.shape
        batch[b, :, :sub_pixel_block_shape[1], :sub_pixel_block_shape[2]] = sub_pixel_block

    return batch, batch_height, batch_width


def batch_to_tile(batch, batch_height, batch_width):
    """
    Convert a batch of mini tiles to a super flat tile
    :param batch: batch numpy array [B, D, H, W]
    :param batch_height: number of tiles vertically
    :param batch_width: number of tiles horizontally
    :return:
    """
    batch_size, bands, inner_height, inner_width = batch.shape
    tile = np.zeros(shape=(bands, inner_height * batch_height, inner_width * batch_width), dtype=batch.dtype)

    for b in range(batch_width * batch_height):
        y = int(b / batch_width)
        x = int(b % batch_width)

        tile[:, y * inner_height: (y+1) * inner_height, x * inner_width:(x+1) * inner_width] = batch[b]

    return tile


def convert_bounding_boxes_to_coord_list(bounding_boxes):
    """
    convert bounding box numpy array to python list of point arrays
    :param bounding_boxes: numpy array of shape [n, 4]
    :return: python array of point numpy arrays, each point array is in shape [4,2]
    """
    num_bounding_boxes = bounding_boxes.shape[0]
    bounding_box_coord_list = []
    for i in range(num_bounding_boxes):
        coord_array = np.empty(shape=(4, 2), dtype=np.float)
        coord_array[0][0] = bounding_boxes[i][0]
        coord_array[0][1] = bounding_boxes[i][1]

        coord_array[1][0] = bounding_boxes[i][0]
        coord_array[1][1] = bounding_boxes[i][3]

        coord_array[2][0] = bounding_boxes[i][2]
        coord_array[2][1] = bounding_boxes[i][3]

        coord_array[3][0] = bounding_boxes[i][2]
        coord_array[3][1] = bounding_boxes[i][1]

        bounding_box_coord_list.append(coord_array)

    return bounding_box_coord_list


def remove_bbox_in_padding(bounding_boxes, image_height, image_width, padding):
    keep_indices = np.where((bounding_boxes[:, 0] < image_height - padding) &
                            (bounding_boxes[:, 1] < image_width - padding) &
                            (bounding_boxes[:, 2] > padding) &
                            (bounding_boxes[:, 3] > padding))

    return bounding_boxes[keep_indices]


def batch_detection_results_to_tile_results(
        batch_bounding_boxes,
        batch_scores,
        batch_classes,
        image_height,
        image_width,
        padding,
        batch_width):
    """
    Convert a batch of object detection results to the format of detections in a flat tile
    :param batch_bounding_boxes: python list (len is batch size) whose element is a [N, 4] numpy array that represents the
                                 bounding box in [y1, x1, y2, x2] with respect to the upper left corner
    :param batch_scores: python list (len is batch size) whose element is a [N,] numpy array that represents the confidence
                         score of each bounding box
    :param batch_classes: python list (len is batch size) whose element is a [N,] numpy array that represents the class
                          of each bounding box
    :param image_height: the model's height
    :param image_width: the model's width
    :param padding: padding in a mini tile
    :param batch_width: number of mini tiles horizontally
    :return: python list of numpy arrays [4,2] that has each point in [y,x], scores numpy array [N,] and class numpy
             array [N,]
    """
    for classes in batch_classes:
        classes = classes.astype(np.uint8)

    inner_width = image_height - 2 * padding
    inner_height = image_width - 2 * padding

    for batch_idx in range(len(batch_bounding_boxes)):
        y, x = int(batch_idx // batch_width), int(batch_idx % batch_width)
        batch_bounding_boxes[batch_idx] = remove_bbox_in_padding(batch_bounding_boxes[batch_idx],
                                                                 image_height, image_width, padding)
        batch_bounding_boxes[batch_idx] = batch_bounding_boxes[batch_idx] - padding
        batch_bounding_boxes[batch_idx][:, [0, 2]] = batch_bounding_boxes[batch_idx][:, [0, 2]] + y * inner_height
        batch_bounding_boxes[batch_idx][:, [1, 3]] = batch_bounding_boxes[batch_idx][:, [1, 3]] + x * inner_width

    bounding_boxes = np.empty(shape=(0, 4))
    scores = np.empty(shape=(0))
    classes = np.empty(shape=(0), dtype=np.uint8)
    for batch_idx in range(len(batch_bounding_boxes)):
        bounding_boxes = np.concatenate((bounding_boxes, batch_bounding_boxes[batch_idx]), axis=0)
        scores = np.concatenate((scores, batch_scores[batch_idx]))
        classes = np.concatenate((classes, batch_classes[batch_idx])).astype(np.uint8)

    return convert_bounding_boxes_to_coord_list(bounding_boxes), scores, classes


def get_available_device(max_memory=0.8):
    '''
    select available device based on the memory utilization status of the device
    :param max_memory: the maximum memory utilization ratio that is considered available
    :return: GPU id that is available, -1 means no GPU is available/uses CPU, if GPUtil package is not installed, will
    return 0 
    '''
    try:
        import GPUtil
    except ModuleNotFoundError:
        return 0

    GPUs = GPUtil.getGPUs()
    freeMemory = 0
    available=-1
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available