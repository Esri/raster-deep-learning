import json
import sys, os, importlib
sys.path.append(os.path.dirname(__file__))

import numpy as np
import math
import arcpy
import cv2



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
    available = 0
    for GPU in GPUs:
        if GPU.memoryUtil > max_memory:
            continue
        if GPU.memoryFree >= freeMemory:
            freeMemory = GPU.memoryFree
            available = GPU.id

    return available

def get_centroid(polygon):
    polygon = np.array(polygon)
    return [polygon[:, 0].mean(), polygon[:, 1].mean()]        

def check_centroid_in_center(centroid, start_x, start_y, chip_sz, padding):
    return ((centroid[1] >= (start_y + padding)) and                  (centroid[1] <= (start_y + (chip_sz - padding))) and                 (centroid[0] >= (start_x + padding)) and                 (centroid[0] <= (start_x + (chip_sz - padding))))

def find_i_j(centroid, n_rows, n_cols, chip_sz, padding, filter_detections):
    for i in range(n_rows):
        for j in range(n_cols):
            start_x = i * chip_sz
            start_y = j * chip_sz

            if (centroid[1] > (start_y)) and (centroid[1] < (start_y + (chip_sz))) and (centroid[0] > (start_x)) and (centroid[0] < (start_x + (chip_sz))):
                in_center = check_centroid_in_center(centroid, start_x, start_y, chip_sz, padding)
                if filter_detections:
                    if in_center: 
                        return i, j, in_center
                else:
                    return i, j, in_center
    return None 

def calculate_rectangle_size_from_batch_size(batch_size):
    """
    calculate number of rows and cols to composite a rectangle given a batch size
    :param batch_size:
    :return: number of cols and number of rows
    """
    rectangle_height = int(math.sqrt(batch_size) + 0.5)
    rectangle_width = int(batch_size / rectangle_height)

    if rectangle_height * rectangle_width > batch_size:
        if rectangle_height >= rectangle_width:
            rectangle_height = rectangle_height - 1
        else:
            rectangle_width = rectangle_width - 1

    if (rectangle_height + 1) * rectangle_width <= batch_size:
        rectangle_height = rectangle_height + 1
    if (rectangle_width + 1) * rectangle_height <= batch_size:
        rectangle_width = rectangle_width + 1

    # swap col and row to make a horizontal rect
    if rectangle_height > rectangle_width:
        rectangle_height, rectangle_width = rectangle_width, rectangle_height

    if rectangle_height * rectangle_width != batch_size:
        return batch_size, 1

    return rectangle_height, rectangle_width
    
def get_tile_size(model_height, model_width, padding, batch_height, batch_width):
    """
    Calculate request tile size given model and batch dimensions
    :param model_height:
    :param model_width:
    :param padding:
    :param batch_width:
    :param batch_height:
    :return: tile height and tile width
    """
    tile_height = (model_height - 2 * padding) * batch_height
    tile_width = (model_width - 2 * padding) * batch_width

    return tile_height, tile_width
    
def tile_to_batch(
    pixel_block, model_height, model_width, padding, fixed_tile_size=True, **kwargs
):
    inner_width = model_width - 2 * padding
    inner_height = model_height - 2 * padding

    band_count, pb_height, pb_width = pixel_block.shape
    pixel_type = pixel_block.dtype

    if fixed_tile_size is True:
        batch_height = kwargs["batch_height"]
        batch_width = kwargs["batch_width"]
    else:
        batch_height = math.ceil((pb_height - 2 * padding) / inner_height)
        batch_width = math.ceil((pb_width - 2 * padding) / inner_width)

    batch = np.zeros(
        shape=(batch_width * batch_height, band_count, model_height, model_width),
        dtype=pixel_type,
    )
    for b in range(batch_width * batch_height):
        y = int(b / batch_width)
        x = int(b % batch_width)

        # pixel block might not be the shape (band_count, model_height, model_width)
        sub_pixel_block = pixel_block[
            :,
            y * inner_height : y * inner_height + model_height,
            x * inner_width : x * inner_width + model_width,
        ]
        sub_pixel_block_shape = sub_pixel_block.shape
        batch[
            b, :, : sub_pixel_block_shape[1], : sub_pixel_block_shape[2]
        ] = sub_pixel_block

    return batch, batch_height, batch_width
    
    
def convert_bounding_boxes_to_coord_list(bounding_boxes):
    """
    convert bounding box numpy array to python list of point arrays
    :param bounding_boxes: numpy array of shape [n, 4]
    :return: python array of point numpy arrays, each point array is in shape [4,2]
    """
    num_bounding_boxes = bounding_boxes.shape[0]
    bounding_box_coord_list = []
    for i in range(num_bounding_boxes):
        coord_array = np.empty(shape=(4, 2), dtype=np.float64)
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
    
 
features = {
    'displayFieldName': '',
    'fieldAliases': {
        'FID': 'FID',
        'Class': 'Class',
        'Confidence': 'Confidence'
    },
    'geometryType': 'esriGeometryPolygon',
    'fields': [
        {
            'name': 'FID',
            'type': 'esriFieldTypeOID',
            'alias': 'FID'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Confidence',
            'type': 'esriFieldTypeDouble',
            'alias': 'Confidence'
        }
    ],
    'features': []
}

fields = {
    'fields': [
        {
            'name': 'OID',
            'type': 'esriFieldTypeOID',
            'alias': 'OID'
        },
        {
            'name': 'Class',
            'type': 'esriFieldTypeString',
            'alias': 'Class'
        },
        {
            'name': 'Confidence',
            'type': 'esriFieldTypeDouble',
            'alias': 'Confidence'
        },
        {
            'name': 'Shape',
            'type': 'esriFieldTypeGeometry',
            'alias': 'Shape'
        }
    ]
}

class GeometryType:
    Point = 1
    Multipoint = 2
    Polyline = 3
    Polygon = 4
    
class GroundingDINO:
    def __init__(self):
        self.name = "GroundingDINO Model"
        self.description = "This python raster function applies computer vision to draw bounding box from text input"
        
    def initialize(self, **kwargs):
        if 'model' not in kwargs:
            return

        model = kwargs['model']
        model_as_file = True
        try:
            with open(model, 'r') as f:
                self.json_info = json.load(f)
        except FileNotFoundError:
            try:
                self.json_info = json.loads(model)
                model_as_file = False
            except json.decoder.JSONDecodeError:
                raise Exception("Invalid model argument")

        
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device = None
        self.device_id = None
        if 'device' in kwargs:
            device = kwargs['device']
            if device == -2:
                device = get_available_device()

        if device is not None:
            if device >= 0:
                try:
                    import torch
                except Exception:
                    raise Exception("PyTorch is not installed. Install it using conda install -c esri deep-learning-essentials")
                torch.cuda.set_device(device)
                arcpy.env.processorType = "GPU"
                arcpy.env.gpuId = str(device)
                self.device_id = device
            else:
                arcpy.env.processorType = "CPU"
                self.device_id = "cpu"
        
        # appending the current dir to path so that groundingdino and supervision can be imported     
        gdino_root_dir = os.path.join(os.path.dirname(__file__), 'GroundingDINO-main')
        supervision_root_dir = os.path.dirname(__file__)

        if gdino_root_dir not in sys.path:
            sys.path.insert(0, gdino_root_dir)
        if supervision_root_dir not in sys.path:
            sys.path.insert(0, supervision_root_dir)
          
        # importing groundingdino and other dependencies
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict
        import torch
        
        
        # loading the GroundingDINO model checkpoint and config file and initliazing GroundingDINO
        cache_file = os.path.join(gdino_root_dir,r"models/groundingdino_swinb_cogcoor.pth")
        cache_config_file = os.path.join(gdino_root_dir,r"models/GroundingDINO_SwinB.cfg.py")        
        args = SLConfig.fromfile(cache_config_file)
        self.groundingdino_model = build_model(args)
        self.groundingdino_model.to(device=self.device_id)
        checkpoint = torch.load(cache_file, map_location="cpu")
        self.groundingdino_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.groundingdino_model.eval()

        
        
            
    def getParameterInfo(self):
        required_parameters = [
            {
                "name": "raster",
                "dataType": "raster",
                "required": True,
                "displayName": "Raster",
                "description": "Input Raster",
            },
            {
                "name": "model",
                "dataType": "string",
                "required": True,
                "displayName": "Input Model Definition (EMD) File",
                "description": "Input model definition (EMD) JSON file",
            },
            {
                "name": "device",
                "dataType": "numeric",
                "required": False,
                "displayName": "Device ID",
                "description": "Device ID",
            },
        ]
        required_parameters.extend(
            [
                {
                    "name": "text_prompt",
                    "dataType": "string",
                    "required": False,
                    "value": "",
                    "displayName": "Text Prompt",
                    "description": "Text Prompt",
                },
                {
                    "name": "padding",
                    "dataType": "numeric",
                    "value": int(self.json_info["ImageHeight"]) // 4,
                    "required": False,
                    "displayName": "Padding",
                    "description": "Padding",
                },
                {
                    "name": "batch_size",
                    "dataType": "numeric",
                    "required": False,
                    "value": 4,
                    "displayName": "Batch Size",
                    "description": "Batch Size",
                },
                {
                    "name": "box_threshold",
                    "dataType": "numeric",
                    "required": False,
                    "value": 0.2,
                    "displayName": "Box Threshold",
                    "description": "Box Threshold",
                },
                {
                    "name": "text_threshold",
                    "dataType": "numeric",
                    "required": False,
                    "value": 0.2,
                    "displayName": "Text Threshold",
                    "description": "Text Threshold",
                },
                {
                    "name": "tta_scales",
                    "dataType": "string",
                    "required": False,
                    "value": "1"
                    if "tta_scales" not in self.json_info
                    else str(self.json_info["tta_scales"]),
                    "displayName": "Perform test time augmentation while predicting using different scales",
                    "description": "provide different scales separated by comma e.g. 0.9,1,1.1",
                },
                {
                    "name": "nms_overlap",
                    "dataType": "numeric",
                    "value": 0.1,
                    "required": False,
                    "displayName": "NMS Overlap",
                    "description": "Maximum allowed overlap within each chip",
                },
                {
                    "name": "exclude_pad_detections",
                    "dataType": "string",
                    "required": False,
                    "domain": ("True", "False"),
                    "value": "True",
                    "displayName": "Filter Outer Padding Detections",
                    "description": "Filter detections which are outside the specified padding",
                },
                
                
                
               
            ]
        )
        return required_parameters
        
        
    def getConfiguration(self, **scalars):
        self.tytx = int(scalars.get("tile_size", self.json_info["ImageHeight"]))
        self.batch_size = (
            int(math.sqrt(int(scalars.get("batch_size", 4)))) ** 2
        )
        self.padding = int(
            scalars.get("padding", self.tytx // 4)
        )
        self.text_prompt =scalars.get("text_prompt")
        
        self.box_threshold = float(
            scalars.get("box_threshold")
        )
        self.text_threshold = float(
            scalars.get("text_threshold")
        )
        (
            self.rectangle_height,
            self.rectangle_width,
        ) = calculate_rectangle_size_from_batch_size(self.batch_size)
        self.tta_scales = scalars.get("tta_scales")
        self.nms_overlap = float(
            scalars.get("nms_overlap", 0.1)
        )
        
        ty, tx = get_tile_size(
            self.tytx,
            self.tytx,
            self.padding,
            self.rectangle_height,
            self.rectangle_width,
        )
        self.exclude_pad_detections = scalars.get(
            "exclude_pad_detections", "True"
        ).lower() in [
            "true",
            "1",
            "t",
            "y",
            "yes",
        ]  ## Default value True
        return {
            "inputMask": True,
            "extractBands": tuple(self.json_info["ExtractBands"]),
            "padding": self.padding,
            "batch_size": self.batch_size,
            "tx": tx,
            "ty": ty,
            "fixedTileSize": 1,
        }
        
        
    def getFields(self):
        return json.dumps(fields)

    def getGeometryType(self):
        return GeometryType.Polygon

    def vectorize(self, **pixelBlocks):
        raster_mask = pixelBlocks["raster_mask"]
        raster_pixels = pixelBlocks["raster_pixels"]
        raster_pixels[np.where(raster_mask == 0)] = 0
        pixelBlocks['raster_pixels'] = raster_pixels
        
        
        scales  = self.tta_scales.split(",")
        bboxes,scores ,labels_f = self.tta_scale_detect_objects(scales,**pixelBlocks)

        
        features['features'] = []
        
        if bboxes.prod():
            polygons = convert_bounding_boxes_to_coord_list(bboxes)
            
            for i in range(len(polygons)):
                rings = [[]]
                for j in range(polygons[i].shape[0]):
                    rings[0].append([polygons[i][j][1], polygons[i][j][0]])
                features['features'].append({
                    'attributes': {
                        'OID': i + 1,
                        'Class': labels_f[i],
                        'Confidence': scores[i]
                    },
                    'geometry': {
                        'rings': rings
                    }
            })
            
        return {"output_vectors": json.dumps(features)}
        
        
        
        
    def detect_objects(self, **pixelBlocks):
        raster_pixels = pixelBlocks["raster_pixels"]
        batch, batch_height, batch_width = tile_to_batch(
            raster_pixels,
            self.tytx,
            self.tytx,
            self.padding,
            fixed_tile_size=True,
            batch_height=self.rectangle_height,
            batch_width=self.rectangle_width,
        )
        from PIL import Image
        import torch
        import groundingdino.datasets.transforms as T
        from groundingdino.util import box_ops
        from groundingdino.util.inference import predict
        finalout = []
        for batch_idx,input_pixels in enumerate(batch):
            input_pixels = np.moveaxis(input_pixels,0,-1)
            pil_image = Image.fromarray(input_pixels)
            transform = T.Compose(
                    [
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )
            image_transformed, _ = transform(pil_image, None)
            final_caption = ""
            if "," in self.text_prompt:
                split_prompts = self.text_prompt.split(',')
                cleaned_items = [split_prompt.strip() for split_prompt in split_prompts]
                final_caption = ' . '.join(cleaned_items)
            else:
                final_caption = self.text_prompt
                
            try:
                boxes, logits, phrases = predict(
                    model=self.groundingdino_model,
                    image=image_transformed,
                    caption=final_caption,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device_id,
                )
            except RuntimeError as e:
                if ("no elements") in str(e):
                    continue
            
            W, H = pil_image.size
            boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
            updated_boxes = []
            updated_scores = []
            updated_phrases = []
            for en1, box1 in enumerate(boxes):
                box1 = box1.numpy()
                x, y, x1, y1 = box1
                w = x1-x
                h = y1 - y
                if w < 0.25*int(self.tytx) and h <int(0.25*self.tytx):
                    updated_boxes.append(boxes[en1].numpy())
                    updated_scores.append(logits[en1].numpy())
                    updated_phrases.append(phrases[en1])
            if updated_boxes:
                output_json = {"boxes":torch.tensor(updated_boxes),"labels":np.array(updated_phrases),"scores":torch.tensor([item.item() for item in updated_scores])}
            else:
                output_json = {"boxes":torch.empty(0, 4),"labels":np.empty(0),"scores":torch.empty(0)}
            
            finalout.append(output_json)
            
        post_processed_pred = []
        for p in finalout:
            bbox, label, score = p["boxes"], p["labels"], p["scores"]
            bbox = bbox / self.tytx
            bbox = torch.index_select(
                        bbox.to(self.device_id), 1, torch.tensor([1, 0, 3, 2]).to(self.device_id)
                    )
            post_processed_pred.append(
                        (bbox, label, score)
                    )

        num_boxes = 0
        for batch_idx in range(self.batch_size):
            num_boxes = num_boxes + post_processed_pred[batch_idx][0].size(0)
            
        bounding_boxes = np.empty(shape=(num_boxes, 4), dtype=float)
        scores = np.empty(shape=(num_boxes), dtype=float)
        classes = np.empty(shape=(num_boxes), dtype=object)
            
        side = math.sqrt(self.batch_size)
        idx = 0
        for batch_idx in range(self.batch_size):
            i, j = batch_idx // side, batch_idx % side

            for bbox, label, score in zip(
                post_processed_pred[batch_idx][0], post_processed_pred[batch_idx][1], post_processed_pred[batch_idx][2]
            ):
        #         bbox = (bbox + 1) / 2
                bounding_boxes[idx, 0] = (bbox.data[0] + i) * self.tytx
                bounding_boxes[idx, 1] = (bbox.data[1] + j) * self.tytx
                bounding_boxes[idx, 2] = (bbox.data[2] + i) * self.tytx
                bounding_boxes[idx, 3] = (bbox.data[3] + j) * self.tytx
                scores[idx] = score.data
                classes[idx] = label
                idx = idx + 1
                
        polygon_list = convert_bounding_boxes_to_coord_list(bounding_boxes)
        score_last = np.array(scores * 100).astype(float)
        label_last = classes
        
        
        padding = self.padding
        keep_polygon = []
        keep_scores = []
        keep_classes = []
        
        chip_sz = self.json_info["ImageHeight"]
        
        for idx, polygon in enumerate(polygon_list):
            centroid = polygon.mean(0)
            i, j = int(centroid[0]) // chip_sz, int(centroid[1]) // chip_sz
            x, y = int(centroid[0]) % chip_sz, int(centroid[1]) % chip_sz

            x1, y1 = polygon[0]
            x2, y2 = polygon[2]

            # fix polygon by removing padded regions
            polygon[:, 0] = polygon[:, 0] - (2 * i + 1) * padding
            polygon[:, 1] = polygon[:, 1] - (2 * j + 1) * padding

            X1, Y1, X2, Y2 = (
                i * chip_sz,
                j * chip_sz,
                (i + 1) * chip_sz,
                (j + 1) * chip_sz,
            )
            t = 2.0  # within 2 pixels of edge

            # if centroid not in center, reduce confidence
            # so box can be filtered out during NMS
            if (
                x < padding
                or x > chip_sz - padding
                or y < padding
                and y > chip_sz - padding
            ):

                score_last[idx] = (self.box_threshold * 100) + score_last[
                    idx
                ] * 0.01

            # if not excluded due to touching edge of tile
            if not (
                self.exclude_pad_detections
                and any(
                    [
                        abs(X1 - x1) < t,
                        abs(X2 - x2) < t,
                        abs(Y1 - y1) < t,
                        abs(Y2 - y2) < t,
                    ]
                )
            ):  # touches edge
                keep_polygon.append(polygon)
                keep_scores.append(score_last[idx])
                keep_classes.append(label_last[idx])
        
        return keep_polygon,keep_scores,keep_classes
        
        
    def tta_scale_detect_objects(self, scales,**pixelBlocks):
        import cv2,torch
        allboxes = torch.empty(0,4)
        allclasses = []
        allscores = torch.empty(0)
        boxes_list, scores_list, labels_list = [], [], []
        
        tile_size = pixelBlocks["raster_pixels"].shape[1]
        pad =  self.padding
        class_mapping = {}
        next_number = 0
        pixel_block = pixelBlocks["raster_pixels"]
        for scale in scales:
            scale = float(scale)
            input_image = pixel_block
            raster_pixels = np.moveaxis(input_image,0,-1)
            rpc = np.moveaxis(input_image,0,-1)

            if scale <= 1:
                padding_scale = 1-scale
                resize_tytx =round(raster_pixels.shape[0]*scale)
                extra_padding_tytx = round(raster_pixels.shape[0]*padding_scale)//2

                resized_image = cv2.resize(raster_pixels, (resize_tytx, resize_tytx), interpolation = cv2.INTER_CUBIC)
                border_img = cv2.copyMakeBorder(resized_image,extra_padding_tytx,extra_padding_tytx,extra_padding_tytx,extra_padding_tytx,borderType=cv2.BORDER_CONSTANT,value=[0, 0, 0,0])
            else:
                padding_scale = scale-1
                resize_tytx =round(raster_pixels.shape[0]*scale)
                original_width = raster_pixels.shape[0]
                extra_padding_tytx = round(resize_tytx - original_width) // 2
                resized_image = cv2.resize(raster_pixels, (resize_tytx, resize_tytx), interpolation = cv2.INTER_CUBIC)
                scaled_width = resized_image.shape[0]
                bottom_coords = extra_padding_tytx + original_width
                border_img = resized_image[extra_padding_tytx:bottom_coords, extra_padding_tytx:bottom_coords]
            raster_pixels = np.moveaxis(border_img,-1,0)

            pixelBlocks["raster_pixels"] = raster_pixels
            polygon_list, scores, classes = self.detect_objects(**pixelBlocks)

            updated_tta_polygons = []
            if scale <= 1:
                for polygons in polygon_list:
                    updated_tta_polygons.append(((polygons+self.padding)-extra_padding_tytx)/scale)  
            else:
                for polygons in polygon_list:
                    updated_tta_polygons.append(((polygons+self.padding)+extra_padding_tytx)*original_width/resize_tytx)
                    
            updated_tta_polygons = np.array(updated_tta_polygons)-(self.padding)*2
                    
            
            mapped_classes, obj_number, next_val = self.map_objects(class_mapping, next_number, classes)
            class_mapping = obj_number
            next_number = next_val

            bboxes  = self.get_img_bbox(tile_size,updated_tta_polygons,scores, mapped_classes)
            
            
            if bboxes is not None:
                allboxes = torch.cat([allboxes, (bboxes.data[0]+1) / 2.0])
                allclasses = allclasses + bboxes.data[1].tolist()
                allscores = np.concatenate([allscores, torch.tensor(scores) * 0.01])
                boxes_list.append((bboxes.data[0] + 1) / 2.0)
                scores_list.append(torch.tensor(scores) * 0.01)
                labels_list.append(bboxes.data[1].tolist())

        try:
            from ensemble_boxes import weighted_boxes_fusion
            iou_thr = self.nms_overlap
            skip_box_thr = 0.0001
            boxes, scores, labels = weighted_boxes_fusion(
                boxes_list,
                scores_list,
                labels_list,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
        except:
            import warnings
            warnings.warn("Unable to perform weighted boxes fusion... use NMS")
            boxes, scores, labels = np.array(allboxes), allscores, np.array(allclasses)
                
        bboxes = boxes * tile_size
        number_to_value = {v: k for k, v in class_mapping.items()}

        if len(labels) > 0:
            labels = np.vectorize(number_to_value.get)(labels)
        
                    
        return bboxes, np.array(scores * 100).astype(float),labels
        
        
        
    def get_img_bbox(self, tile_size, polygons, scores, classes):
        from fastai.vision import ImageBBox

        pad = self.padding
        bboxes = []
        for i, polygon in enumerate(polygons):
            x1, y1 = np.around(polygon).astype(int)[0]
            x2, y2 = np.around(polygon).astype(int)[2]
            bboxes.append([x1 + pad, y1 + pad, x2 + pad, y2 + pad])
        n = len(bboxes)
        if n > 0:
            return ImageBBox.create(
                tile_size,
                tile_size,
                bboxes,
                labels=classes,
                classes=["Background"]*(max(classes)+1)
            
            )
        else:
            return None
            
    def map_objects(self, obj_to_num,next_number, object_list):
    
        result = []

        for obj in object_list:
            if obj not in obj_to_num:
                obj_to_num[obj] = next_number
                next_number += 1
            result.append(obj_to_num[obj])

        return result, obj_to_num, next_number 
                
 
 