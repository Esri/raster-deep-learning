import math
import os

import matplotlib.cm as cmx
import matplotlib.colors as mcolors
import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
from arcgis.geometry import project
from matplotlib import patches, patheffects
from model import V, A, T, to_np
from torch.autograd import Variable

# torch.cuda.set_device(0)

categories = ['ground','pool']
id2cat = categories

anc_grids = [4,2,1]
anc_zooms = [0.7, 1., 1.3]
anc_ratios = [(1.,1.), (1.,0.5), (0.5,1.)]

anchor_scales = [(anz*i,anz*j) for anz in anc_zooms for (i,j) in anc_ratios]
k = len(anchor_scales)
anc_offsets = [1/(o*2) for o in anc_grids]


anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                        for ao,ag in zip(anc_offsets,anc_grids)])
anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                        for ao,ag in zip(anc_offsets,anc_grids)])
anc_ctrs = np.repeat(np.stack([anc_x,anc_y], axis=1), k, axis=0)

anc_sizes  =   np.concatenate([np.array([[o/ag,p/ag] for i in range(ag*ag) for o,p in anchor_scales])
               for ag in anc_grids])
grid_sizes = V(np.concatenate([np.array([ 1/ag       for i in range(ag*ag) for o,p in anchor_scales])
               for ag in anc_grids]), requires_grad=False).unsqueeze(1)
anchors = V(np.concatenate([anc_ctrs, anc_sizes], axis=1), requires_grad=False).float()

def hw2corners(ctr, hw): 
    return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)
anchor_cnr = hw2corners(anchors[:,:2], anchors[:,2:])


# padding = (112, 112, 112, 112)
def get_cropped_tiles(img, border=112):
    new_im = ImageOps.crop(Image.fromarray(img), border=border)
    paddedimg = np.array(new_im)
    padded_tiles = get_tile_images(paddedimg)
    return padded_tiles

def get_img(filename):
    return np.array(Image.open(filename))

def export_img(ext, path, filename, naipfalse):
    try:
        os.mkdir(path)
    except FileExistsError:
        print('path exists! continuing..')
    
    imgpath = naipfalse.export_image(ext, image_sr={'wkid': 4326}, 
                                     bbox_sr=3857, size=[224*8, 224*8], f='image', 
                                     export_format='jpg', adjust_aspect_ratio=False,
                          save_folder=path,
                          save_file=filename)
    
def get_tile_images(image, width=224, height=224):
    _nrows, _ncols, depth = image.shape
    _size = image.size
    _strides = image.strides

    nrows, _m = divmod(_nrows, height)
    ncols, _n = divmod(_ncols, width)
    if _m != 0 or _n != 0:
        return None

    tiles = np.lib.stride_tricks.as_strided(
        np.ravel(image),
        shape=(nrows, ncols, height, width, depth),
        strides=(height * _strides[0], width * _strides[1], *_strides),
        writeable=False)
    
    return tiles.reshape(-1, *tiles.shape[2:])


    

def get_y(bbox,clas):
    bbox = bbox.view(-1,4)/sz
    try:
        xx = ((bbox[:,2]-bbox[:,0])>0).nonzero()
        #print(xx)
        bb_keep = xx[:,0]
    except:
        # print('No bboxes')
        return None, None
    
def actn_to_bb(actn, anchors, grid_sizes):
    actn_bbs = torch.tanh(actn)
    actn_centers = (actn_bbs[:,:2]/2 * grid_sizes) + anchors[:,:2]
    actn_hw = (actn_bbs[:,2:]/2+1) * anchors[:,2:] # [-1, 1]/2 + 1 = [0.5, 1.5] * anchor ht, w
    return hw2corners(actn_centers, actn_hw)    

def pred2dict(bb_np, score, cat_str):
    # convert to top left x,y bottom right x,y
    return {"x1": bb_np[1],
            "x2": bb_np[3],
            "y1": bb_np[0],
            "y2": bb_np[2],
            "score": score,
            "category": cat_str}

def nms(boxes, scores, overlap=0.5, top_k=100):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0: return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1: break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

def nms_preds(a_ic, p_cl, cl):
    nms_bb, nms_pr, nms_id = [],[],[]
    
    conf_scores = p_cl.sigmoid()[0].t().data
    boxes = a_ic.view(-1, 4)
    scores = conf_scores[cl]
    
    if len(scores)>0:
        ids, count = nms(boxes.data, scores, 0.1, 50)
        ids = ids[:count]

        nms_pr.append(scores[ids])
        nms_bb.append(boxes.data[ids])
        nms_id.append([cl]*count)

    else: nms_bb, nms_pr, nms_id = [[-1.,-1.,-1.,-1.,]],[[-1]],[[-1]]
    
    # return in order of a_ic, clas id, clas_pr
    return Variable(torch.cuda.FloatTensor(nms_bb[0])), Variable(torch.cuda.FloatTensor(nms_pr[0])), np.asarray(nms_id[0])


def predictions(bbox, clas=None, prs=None, thresh=0.3):
    #bb = [bb_hw(o) for o in bbox.reshape(-1,4)]
    bb = bbox
    if prs is None:  prs  = [None]*len(bb)
    if clas is None: clas = [None]*len(bb)
    predictions = []
    for i, (b, c, pr) in enumerate(zip(bb, clas, prs)):
        if((b[2]>0) and (pr is None or pr > thresh)):
            cat_str = 'pool' #'bg' if c[i]==len(id2cat) else id2cat[c[i]]
            score = pr
            bb_np = to_np(b).astype('float64')
            predictions.append(pred2dict(bb_np, score, cat_str))
    return predictions
            
def get_nms_preds(b_clas, b_bb, idx, anchors, score_threshold=0.6):
    
    a_ic = actn_to_bb(b_bb[idx], anchors, grid_sizes)
    clas_pr, clas_ids = b_clas[idx].max(1)
    clas_pr = clas_pr.sigmoid()

    conf_scores = b_clas[idx].sigmoid().t().data

    out1, out2, cc = [], [], []
    for cl in range(0, len(conf_scores)-1):
        c_mask = conf_scores[cl] > score_threshold
        if c_mask.sum() == 0: continue
        scores = conf_scores[cl][c_mask]
        l_mask = c_mask.unsqueeze(1).expand_as(a_ic)
        boxes = a_ic[l_mask].view(-1, 4)
        ids, count = nms(boxes.data, scores, 0.1, 50)
        ids = ids[:count]
        out1.append(scores[ids])
        out2.append(boxes.data[ids])
        cc.append([cl]*count)
    if cc == []:
        cc = [[0]]
    cc = T(np.concatenate(cc))
    if out1 == []:
        out1 = [torch.Tensor()]
    out1 = torch.cat(out1)
    if out2 == []:
        out2 = [torch.Tensor()]
    out2 = torch.cat(out2)
    bbox, clas, prs, thresh = out2, cc, out1, 0.
    return predictions(to_np(bbox),
         to_np(clas), to_np(prs) if prs is not None else None, thresh)
    

        
def show_nmf(images, b_clas, b_bb, idx, anchors, ax1):
    ax1.set_axis_off()
    if not len(images.shape) > 3:
        images = images[np.newaxis,:]
    ima = denorm(images[idx]).astype(int)
    
    a_ic = actn_to_bb(b_bb[idx], anchors, grid_sizes)
    clas_pr, clas_ids = b_clas[idx].max(1)
    clas_pr = clas_pr.sigmoid()

    conf_scores = b_clas[idx].sigmoid().t().data

    out1, out2, cc = [], [], []
    for cl in range(0, len(conf_scores)-1):
        c_mask = conf_scores[cl] > 0.1
        if c_mask.sum() == 0: continue
        scores = conf_scores[cl][c_mask]
        l_mask = c_mask.unsqueeze(1).expand_as(a_ic)
        boxes = a_ic[l_mask].view(-1, 4)
        ids, count = nms(boxes.data, scores, 0.1, 50)
        ids = ids[:count]
        out1.append(scores[ids])
        out2.append(boxes.data[ids])
        cc.append([cl]*count)
    if cc == []:
        cc = [[0]]
    cc = T(np.concatenate(cc))
    if out1 == []:
        out1 = [torch.Tensor()]
    out1 = torch.cat(out1)
    if out2 == []:
        out2 = [torch.Tensor()]
    out2 = torch.cat(out2)
    torch_gt(ax1, ima, out2, cc, out1, 0.)

def torch_gt(ax, ima, bbox, clas, prs=None, thresh=0.4):
    return show_ground_truth(ax, ima, to_np((bbox*224).long()),to_np(clas), to_np(prs) if prs is not None else None, thresh)

def show_ground_truth(ax, im, bbox, clas=None, prs=None, thresh=0.3):
    bb = [bb_hw(o) for o in bbox.reshape(-1,4)]
    if prs is None:  prs  = [None]*len(bb)
    if clas is None: clas = [None]*len(bb)
    ax = show_img(im, ax=ax)
    for i,(b,c,pr) in enumerate(zip(bb, clas, prs)):
        if((b[2]>0) and (pr is None or pr > thresh)):
            draw_rect(ax, b, color=colr_list[i%num_colr])
            txt = f'{i}: '
#             print(c)
            if c is not None: txt += ('bg' if c==len(id2cat) else id2cat[c])
            if pr is not None: txt += f' {pr:.2f}'
            draw_text(ax, b[:2], txt, color=colr_list[i%num_colr])
            
def bb_hw(a): return np.array([a[1],a[0],a[3]-a[1]+1,a[2]-a[0]+1])

def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.set_xticks(np.linspace(0, 224, 8))
    ax.set_yticks(np.linspace(0, 224, 8))
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    return ax

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_rect(ax, b, color='white'):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor=color, lw=2))
    draw_outline(patch, 4)

def draw_text(ax, xy, txt, sz=14, color='white'):
    text = ax.text(*xy, txt,
        verticalalignment='top', color=color, fontsize=sz, weight='bold')
    draw_outline(text, 1)
    
def get_cmap(N):
    color_norm  = mcolors.Normalize(vmin=0, vmax=N-1)
    return cmx.ScalarMappable(norm=color_norm, cmap='Set3').to_rgba


num_colr = 12
cmap = get_cmap(num_colr)
colr_list = [cmap(float(x)) for x in range(num_colr)]

def load_model(m, p):
    sd = torch.load(p, map_location=lambda storage, loc: storage)
    m.load_state_dict(sd)
    return m

def predict_(model, images):
    images = V(images)
    if not len(images.size()) > 3:
        images.data = images.data.view(1, *images.size())
    bbox, clas = model(images)
    return bbox, clas

imagenet_stats = A([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

mean = 255* np.array(imagenet_stats[0], dtype=np.float32)
std  = 255* np.array(imagenet_stats[1], dtype=np.float32)

norm = lambda x: (x-mean)/ std
denorm = lambda x: x * std + mean

def find_objects(model, tiles, extent, crop=(0, 0, 0, 0), cycle=1):
    xmin = extent['xmin']  + crop[0]
    ymin = extent['ymin']  + crop[1]
    xmax = extent['xmax']  - crop[2]
    ymax = extent['ymax']  - crop[3]
    
    img_normed = norm(tiles)

    clas, bbox = predict_(model, img_normed.transpose(0, 3, 1, 2))
    preds = { }

    for idx in range(bbox.size()[0]): 
        preds[idx] = get_nms_preds(clas, bbox, idx, anchors)
    
    w, h = 90, 90
    pools = []
    numboxes = bbox.size()[0]
    side = math.sqrt(numboxes)

    for idx in range(numboxes): 
        i, j = idx//side, idx%side

        x = xmin + (j)*w
        y = ymax - (i+1)*h

        for pred in preds[idx]:
            objx = x + ((pred['x1'] + pred['x2']) / 2)*w
            objy = y + h - ((pred['y1'] + pred['y2']) / 2)*h
            pools.append({'x': objx, 'y': objy, 'category': pred['category'], 'score': pred['score']})

    if len(pools) > 0:
        result = project(geometries=pools, in_sr=3857, out_sr=4326)
        for i, p in enumerate(pools):
            p['SHAPE'] = dict(result[i])
            p['cycle']  = cycle

    return pools

def detect_objects_image_space(model, tiles, score_threshold=0.6):
    tile_height, tile_width = tiles.shape[2], tiles.shape[3]
    img_normed = norm(tiles.transpose(0,2,3,1))

    clas, bbox = predict_(model, img_normed.transpose(0, 3, 1, 2))
    preds = { }

    batch_size = tiles.shape[0]

    batch_bounding_boxes, batch_scores, batch_classes = [], [], []
    for batch_idx in range(batch_size):
        pred = get_nms_preds(clas, bbox, batch_idx, anchors, score_threshold)
        num_boxes = len(pred)
        bounding_boxes = np.empty(shape=(num_boxes, 4))
        scores = np.empty(shape=(num_boxes))
        classes = np.empty(shape=(num_boxes), dtype=np.uint8)
        for box_idx in range(num_boxes):
            bounding_boxes[box_idx, 0] = pred[box_idx]['y1']*tile_height
            bounding_boxes[box_idx, 1] = pred[box_idx]['x1']*tile_width
            bounding_boxes[box_idx, 2] = pred[box_idx]['y2']*tile_height
            bounding_boxes[box_idx, 3] = pred[box_idx]['x2']*tile_width

            scores[box_idx] = pred[box_idx]['score']
            classes[box_idx] = 1

        batch_bounding_boxes.append(bounding_boxes)
        batch_scores.append(scores)
        batch_classes.append(classes)

    return batch_bounding_boxes, batch_scores, batch_classes


def detect_objects(model, tiles, extent, crop=(0, 0, 0, 0), cycle=1):
    xmin = extent['xmin']  + crop[0]
    ymin = extent['ymin']  + crop[1]
    xmax = extent['xmax']  - crop[2]
    ymax = extent['ymax']  - crop[3]
    
    img_normed = norm(tiles)

    clas, bbox = predict_(model, img_normed.transpose(0, 3, 1, 2))
    preds = { }

    for idx in range(bbox.size()[0]): 
        preds[idx] = get_nms_preds(clas, bbox, idx, anchors)
    
    w, h = 90, 90
    pools = []
    numboxes = bbox.size()[0]
    side = math.sqrt(numboxes)

    for idx in range(numboxes): 
        i, j = idx//side, idx%side

        x = xmin + (j)*w        # 0 + j*224
        y = ymax - (i+1)*h

        for pred in preds[idx]:
            objx = x + ((pred['x1'] + pred['x2']) / 2)*w
            objy = y + h - ((pred['y1'] + pred['y2']) / 2)*h
            pools.append({'x': objx, 'y': objy, 'category': pred['category'], 'score': pred['score']})

    if len(pools) > 0:
        #result = project(geometries=pools, in_sr=3857, out_sr=4326)
        for i, p in enumerate(pools):
            #p['SHAPE'] = dict(result[i])
            p['cycle'] = cycle

    return pools

def get_distances(coord, other_x_coords, other_y_coords):
    x = coord['x']
    y = coord['y']
    
    distances = (((x - other_x_coords) ** 2) + ((y - other_y_coords) ** 2)) ** 0.5
    return distances

def suppress_close_pools(sdf, min_dist=15, accumulate_thres=88):
    sdf['c_score'] = 0 # vanilla accumulation of scores
    sdf['c_score_b'] = 0 # doesn't let others accumulate already accumulated scores in done_index
    sdf['c_score_c'] = 0 # only lets score above specific threshold(accumulate_thres) accumulate scores from done_index
    sdf['c_count'] = 0 # count of vanilla accumulation scores
    sdf['c_count_b'] = 0 # count of b type accumulation scores
    sdf['c_count_c'] = 0 # count of c type accumulation scores
    
    sorted_sdf = sdf.sort_values(by='score', ascending=False)
    sorted_sdf = sorted_sdf.reset_index().drop('index', axis=1)
    
    # main nms
    selected_idxs = []
    suppressed_idxs = []
    done_index = np.array([0] * sorted_sdf.shape[0])

    for index, row in sorted_sdf.iterrows():
        if done_index[index] == 0:
            dists = get_distances(row, sorted_sdf['x'], sorted_sdf['y'])
            idxs = sorted_sdf[dists < min_dist].index.tolist()
            selected_idxs.append(index) #idxs[0]) 
            
            cumulative_score = sorted_sdf.iloc[idxs].score.sum()
            idxs_for_b = [i for i in idxs if i not in sorted_sdf[done_index == 1].index]
            idxs_for_c = [i for i in idxs if (i not in sorted_sdf[done_index == 1].index) or (sorted_sdf.loc[index,'score'] > accumulate_thres)]
            # print(idxs_for_b, idxs_for_c, idxs)
            
            cumulative_score_b = sorted_sdf.iloc[idxs_for_b].score.sum() 
            cumulative_score_c = sorted_sdf.iloc[idxs_for_c].score.sum() 
            
            to_suppress = [item for item in idxs if item != index]
            
            suppressed_idxs.extend(to_suppress) #idxs[1:])
            
            done_index[idxs] = 1
            
            sorted_sdf.loc[index,'c_score'] = cumulative_score 
            sorted_sdf.loc[index,'c_count'] = len(idxs) # earlier this was len(idxs - 1)
            sorted_sdf.loc[index,'c_score_b'] = cumulative_score_b 
            sorted_sdf.loc[index,'c_count_b'] = len(idxs_for_b)
            sorted_sdf.loc[index,'c_score_c'] = cumulative_score_c 
            sorted_sdf.loc[index,'c_count_c'] = len(idxs_for_c)            
            
    return sorted_sdf.iloc[selected_idxs], sorted_sdf.iloc[suppressed_idxs]
    
def predict_classf(model, images):
    images = V(images)
    if not len(images.size()) > 3:
        images.data = images.data.view(1, *images.size())
    clas = model(images)
    return torch.exp(clas)[:,1]

def overlap(a, b): 
    dx = min(a['xmax'], b['xmax']) - max(a['xmin'], b['xmin'])
    dy = min(a['ymax'], b['ymax']) - max(a['ymin'], b['ymin'])
    if (dx>=0) and (dy>=0):
        return True
    return False