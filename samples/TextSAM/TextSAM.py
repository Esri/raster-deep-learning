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
    
class TextSAM:
    def __init__(self):
        self.name = "Text SAM Model"
        self.description = "This python raster function applies computer vision to segment anything from text input"
        
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
        
        # appending the current dir to path so that segment_anything,groundingdino and supervision can be imported     
        sam_root_dir = os.path.join(os.path.dirname(__file__), 'segment-anything')
        gdino_root_dir = os.path.join(os.path.dirname(__file__), 'GroundingDINO-main')
        supervision_root_dir = os.path.dirname(__file__)
        if sam_root_dir not in sys.path:
            sys.path.insert(0, sam_root_dir)
        if gdino_root_dir not in sys.path:
            sys.path.insert(0, gdino_root_dir)
        if supervision_root_dir not in sys.path:
            sys.path.insert(0, supervision_root_dir)
          
        # importing segment_anything, groundingdino and other dependencies
        from segment_anything import sam_model_registry, SamPredictor
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict
        import torch
        
        # loading the SAM model checkpoint and initliazing SAM mask_generator
        sam_checkpoint =  os.path.join(sam_root_dir, "models/sam_vit_b_01ec64.pth")
        model_type = "vit_b"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device_id)
        self.mask_generator = SamPredictor(sam)
        
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
                    "name": "box_nms_thresh",
                    "dataType": "numeric",
                    "required": False,
                    "value": 0.7,
                    "displayName": "box_nms_thresh",
                    "description": "The box IoU cutoff used by non-maximal suppression to filter duplicate masks.",
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
        ty, tx = get_tile_size(
            self.tytx,
            self.tytx,
            self.padding,
            self.rectangle_height,
            self.rectangle_width,
        )
        
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
        
        # create batch from pixel blocks
        batch, batch_height, batch_width = tile_to_batch(
            raster_pixels,
            self.tytx,
            self.tytx,
            self.padding,
            fixed_tile_size=True,
            batch_height=self.rectangle_height,
            batch_width=self.rectangle_width,
        )
        
        mask_list = []
        score_list = []

        from PIL import Image
        import torch
        import groundingdino.datasets.transforms as T
        from groundingdino.util import box_ops
        from groundingdino.util.inference import predict
        
        # iterate over batch and get segment from model
        for batch_idx,input_pixels in enumerate(batch):
            side = int(math.sqrt(self.batch_size))
            i, j = batch_idx // side, batch_idx % side
            input_pixels = np.moveaxis(input_pixels,0,-1)
            # for input_pixels in batch:
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
            for en1, box1 in enumerate(boxes):
                box1 = box1.numpy()
                x, y, x1, y1 = box1
                w = x1-x
                h = y1 - y
                if w < 0.25*int(self.tytx) and h <int(0.25*self.tytx):
                    updated_boxes.append(boxes[en1])
                    updated_scores.append(logits[en1])
            self.mask_generator.set_image(input_pixels)
            
            if updated_boxes:
                transformed_boxes = self.mask_generator.transform.apply_boxes_torch(
                    torch.stack(updated_boxes), input_pixels.shape[:2]
                    )
                masks, _, _ = self.mask_generator.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes.to(self.device_id),
                    multimask_output=False,
                    )
                for counter, mask_value in enumerate(masks):
                    masked_image = mask_value*1
                    masked_image = masked_image.cpu().numpy()
                    contours, hierarchy = cv2.findContours((masked_image[0]).astype(np.uint8),
                                                            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
                                                            offset=(0, 0))
                    hierarchy = hierarchy[0]
                    for c_idx, contour in enumerate(contours):
                        contour = contours[c_idx] = contour.squeeze(1)
                        contours[c_idx][:, 0] = contour[:, 0] + (j * self.tytx)
                        contours[c_idx][:, 1] = contour[:, 1] + (i * self.tytx)
                    for (contour_idx,(next_contour, prev_contour, child_contour, parent_contour),) in enumerate(hierarchy):
                        if parent_contour == -1:
                            coord_list = [contours[contour_idx].tolist()]
                            while child_contour != -1:
                                coord_list.append(contours[child_contour].tolist())
                                child_contour = hierarchy[child_contour][0]
                            mask_list.append(coord_list)
                            score_list.append(str(updated_scores[counter].numpy()*100))
                
        n_rows = int(math.sqrt(self.batch_size))
        n_cols = int(math.sqrt(self.batch_size))
        padding = self.padding
        keep_masks = []
        keep_scores = []
       
        for idx, mask in enumerate(mask_list):
            if mask == []:
                continue
            centroid = get_centroid(mask[0]) 
            tytx = self.tytx
            grid_location = find_i_j(centroid, n_rows, n_cols, tytx, padding, True)
            if grid_location is not None:
                i, j, in_center = grid_location
                for poly_id, polygon in enumerate(mask):
                    polygon = np.array(polygon)
                    polygon[:, 0] = polygon[:, 0] - (2*i + 1)*padding  # Inplace operation
                    polygon[:, 1] = polygon[:, 1] - (2*j + 1)*padding  # Inplace operation            
                    mask[poly_id] = polygon.tolist()
                if in_center:
                    keep_masks.append(mask)
                    keep_scores.append(score_list[idx])
            
        final_masks =  keep_masks
        pred_score = keep_scores 

        
        features['features'] = []
        
        for mask_idx, final_mask in enumerate(final_masks):
            features['features'].append({
                'attributes': {
                    'OID': mask_idx + 1,
                    'Class': "Segment",
                    'Confidence': pred_score[mask_idx]
                },
                'geometry': {
                    'rings': final_mask
                }
        })
        return {"output_vectors": json.dumps(features)}