from textwrap import wrap
from matplotlib.backends.backend_pdf import PdfPages
from utils import img_heat_bbox_disp
from skimage.feature import peak_local_max
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import sys


def validate_vg(dict_test,txn, num_tst,heatmap_w,heatmap_s, R_i, R_s, lvl_idx_wrd, lvl_idx_sen, lvl_scr_wrd, sess):
    cnt_overall = 0
    cnt_correct_w = 0
    cnt_correct_hit_w = 0
    cnt_correct_s = 0
    cnt_correct_hit_s = 0
    cnt_correct_att_s = 0
    cnt_correct_att_w = 0
    
    wrd_idx_list = []
    sen_idx_list = []
    for k,doc_id in enumerate(dict_test):
        if k>num_tst:
            continue
        imgbin = txn.get(doc_id.encode('utf-8'))
        if imgbin==None:
            continue
        buff = np.frombuffer(imgbin, dtype='uint8')
        if len(buff) == 0:
            continue
        imgbgr = cv2.imdecode(buff, cv2.IMREAD_COLOR)
        imgrgb = imgbgr[:,:,[2,1,0]]

        img = np.reshape(cv2.resize(imgrgb,(299,299)),(1,299,299,3))
        orig_img_shape = dict_test[doc_id]['size'][:2]
        
        sen_batch = []
        annot_batch = []
        #create batch of queries in a doc_id annotation
        for i,annot in enumerate(dict_test[doc_id]['annotations']):
            sen = annot['query']
            if len(sen.split())==0 or len(annot['bbox_norm'])== 0:
                continue
            if not check_percent(union(annot['bbox_norm'])):
                continue
            if any(b>1 for b in annot['bbox_norm']):
                continue
            sen_batch.append(sen)
            annot_batch.append(annot)
        if len(sen_batch)==0:
            continue
        cnt_overall += len(sen_batch)

        img_batch = np.repeat(img,len(sen_batch),axis=0)
        tensor_list = [heatmap_w,heatmap_s, R_i, R_s, lvl_idx_wrd, lvl_idx_sen, lvl_scr_wrd]
        feed_dict = {'input_img:0': img_batch, 'text_input:0': sen_batch, "mode:0": 'test'}
        qry_heats,qry_heat, qry_scores, sen_score, wrd_idx, sen_idx, lvl_scores = sess.run(tensor_list, feed_dict)
        sen_idx_list.extend(sen_idx) 
        
        for c,sen in enumerate(sen_batch):
            idx = [j for j in range(len(sen.split()))]
            wrds = sen.split()
            wrd_idx_list.extend(wrd_idx[c,:len(wrds)])
            if np.mean(qry_scores[c,idx])==0:
                pred = {}
            else:
                heatmap_wrd = np.average(qry_heats[c,idx,:], weights = qry_scores[c,idx], axis=0)
                heatmap_sen = qry_heat[c,:]
                bbox_c_w,hit_c_w,att_c_w = calc_correctness(annot_batch[c],heatmap_wrd,orig_img_shape)
                bbox_c_s,hit_c_s,att_c_s = calc_correctness(annot_batch[c],heatmap_sen,orig_img_shape)
                cnt_correct_w += bbox_c_w
                cnt_correct_hit_w += hit_c_w
                cnt_correct_s += bbox_c_s
                cnt_correct_hit_s += hit_c_s
                cnt_correct_att_w += att_c_w 
                cnt_correct_att_s += att_c_s

        var = [k,num_tst,cnt_correct_w/cnt_overall,cnt_correct_hit_w/cnt_overall]
        var_s = [cnt_correct_s/cnt_overall,cnt_correct_hit_s/cnt_overall]
        prnt0 = 'Sample {}/{}, IoU_acc_w:{:.2f}, IoU_acc_s:{:.2f}'.format(var[0],var[1],var[2],var_s[0])
        prnt1 = ', Hit_acc_w:{:.2f}, Hit_acc_s:{:.2f} \r'.format(var[3],var_s[1])
        sys.stdout.write(prnt0+prnt1)                
        sys.stdout.flush()

    hit_acc_w = cnt_correct_hit_w/cnt_overall
    iou_acc_w = cnt_correct_w/cnt_overall
    hit_acc_s = cnt_correct_hit_s/cnt_overall
    iou_acc_s = cnt_correct_s/cnt_overall
    att_acc_w = cnt_correct_att_w/cnt_overall
    att_acc_s = cnt_correct_att_s/cnt_overall
    
    return iou_acc_w, hit_acc_w,att_acc_w,iou_acc_s, hit_acc_s, att_acc_s




def validate_referit(dict_test,txn,num_tst, heatmap_w,heatmap_s, R_i, R_s, lvl_idx_wrd, lvl_idx_sen, lvl_scr_wrd,sess):
    cnt_overall = 0
    cnt_correct_w = 0
    cnt_correct_hit_w = 0
    cnt_correct_s = 0
    cnt_correct_hit_s = 0
    cnt_correct_att_s = 0
    cnt_correct_att_w = 0
    wrd_idx_list = []
    sen_idx_list = []
    for k,doc_id in enumerate(dict_test):
        if k>num_tst:
            continue
        imgbin = txn.get(doc_id.encode('utf-8'))
        if imgbin==None:
            print ("Image not found")
            continue
        buff = np.frombuffer(imgbin, dtype='uint8')
        if len(buff) == 0:
            print ("Image not found")
            continue
        imgbgr = cv2.imdecode(buff, cv2.IMREAD_COLOR)
        imgrgb = imgbgr[:,:,[2,1,0]]

        img = np.reshape(cv2.resize(imgrgb,(299,299)),(1,299,299,3))
        orig_img_shape = dict_test[doc_id]['size'][:2]

        for i,annot in enumerate(dict_test[doc_id]['annotations']):
            if len(annot['bbox_norm'])== 0:
                continue
            if not check_percent(union(annot['bbox_norm'])):
                continue
            if any(b>1 for b in annot['bbox_norm']):
                continue
            unq_qry = set(annot['query'])
            sen_batch = [sen for sen in unq_qry if 0<len(sen.split())<=50] #only unique queries with 0<length<=50
            img_batch = np.repeat(img,len(sen_batch),axis=0)
            tensor_list = [heatmap_w, heatmap_s,R_i, R_s, lvl_idx_wrd, lvl_idx_sen, lvl_scr_wrd]
            feed_dict = {'input_img:0': img_batch, 'text_input:0': sen_batch, 'mode:0': 'test'}
            
            qry_heats, qry_heat, qry_scores, sen_score, wrd_idx, sen_idx, lvl_scores = sess.run(tensor_list, feed_dict)
            sen_idx_list.extend(sen_idx)
            cnt_overall += len(sen_batch)
            for c,sen in enumerate(sen_batch):
                idx = [j for j in range(len(sen.split()))]
                if np.mean(qry_scores[c,idx])==0:
                    pred = {}
                else:
                    wrds = sen.split()
                    wrd_idx_list.extend(wrd_idx[c,:len(wrds)])
                    heatmap_wrd = np.average(qry_heats[c,idx,:], weights = qry_scores[c,idx], axis=0)
                    heatmap_sen = qry_heat[c,:]
                    bbox_c_w, hit_c_w, att_c_w = calc_correctness(annot,heatmap_wrd,orig_img_shape)
                    bbox_c_s, hit_c_s, att_c_s = calc_correctness(annot,heatmap_sen,orig_img_shape)
                    cnt_correct_w += bbox_c_w
                    cnt_correct_hit_w += hit_c_w
                    cnt_correct_s += bbox_c_s
                    cnt_correct_hit_s += hit_c_s
                    cnt_correct_att_w += att_c_w 
                    cnt_correct_att_s += att_c_s
                                     
        var = [k,num_tst,cnt_correct_w/cnt_overall,cnt_correct_hit_w/cnt_overall,cnt_correct_att_w /cnt_overall,cnt_correct_att_s/cnt_overall  ]
        var_s = [cnt_correct_s/cnt_overall,cnt_correct_hit_s/cnt_overall]
        prnt0 = 'Sample {}/{}, IoU_acc_w:{:.2f}, IoU_acc_s:{:.2f}'.format(var[0],var[1],var[2],var_s[0])
        prnt1 = ', Hit_acc_w:{:.2f}, Hit_acc_s:{:.2f} \r'.format(var[3],var_s[1])
        prnt2 = ', Att_acc_w:{:.2f}, Att_acc_s:{:.2f} \r'.format(var[4],var[5])
        
        sys.stdout.write(prnt0+prnt1)                
        sys.stdout.flush()

    hit_acc_w = cnt_correct_hit_w/cnt_overall
    iou_acc_w = cnt_correct_w/cnt_overall
    hit_acc_s = cnt_correct_hit_s/cnt_overall
    iou_acc_s = cnt_correct_s/cnt_overall
    att_acc_w = cnt_correct_att_w/cnt_overall
    att_acc_s = cnt_correct_att_s/cnt_overall
    
    return iou_acc_w, hit_acc_w,att_acc_w,iou_acc_s, hit_acc_s, att_acc_s



def validate_flickr30k(dict_test,num_tst, heatmap_w, R_i, R_s,sess):
    
    cnt_overall = 0
    cnt_correct = 0
    cnt_correct_hit = 0
    cnt_correct_att = 0
    
    for k, doc_id in enumerate(dict_test):
        if k > num_tst:
            continue
        img = np.reshape(cv2.resize(dict_test[doc_id]['img'], (299, 299)), (1, 299, 299, 3))
        orig_img_shape = dict_test[doc_id]['size']
        sen_batch = list(dict_test[doc_id]['sentences'].keys())
        img_batch = np.repeat(img, len(sen_batch), axis=0)
        tensor_list = [heatmap_w, R_i, R_s]
        feed_dict = {'input_img:0': img_batch, 'text_input:0': sen_batch, 'mode:0': 'test'}
        qry_heats, qry_scores, sen_score = sess.run(tensor_list, feed_dict)
        
        for c, sen in enumerate(sen_batch):
            for query in dict_test[doc_id]['sentences'][sen]:
                # reject not groundable/acceptable queries
                idx = dict_test[doc_id]['sentences'][sen][query]['idx']
                if len(query.split()) == 0 or len(idx) == 0:
                    continue
                annot = dict_test[doc_id]['sentences'][sen][query]
                category = annot['category']
                if 'notvisual' in category or len(annot['bbox_norm']) == 0:
                    continue
                if not check_percent(union(annot['bbox_norm'])):
                    continue
                # if reaches this point, it is groundable/acceptable
                cnt_overall += 1

                if np.mean(qry_scores[c, idx]) == 0:
                    pred = {}
                else:
                    heatmap = np.average(qry_heats[c, idx, :], weights=qry_scores[c, idx], axis=0)
                    bbox_c, hit_c, att_c = calc_correctness(annot, heatmap, orig_img_shape)
                    cnt_correct_att += att_c
                    cnt_correct += bbox_c
                    cnt_correct_hit += hit_c

        var = [k, num_tst, cnt_correct / cnt_overall, cnt_correct_hit / cnt_overall, cnt_correct_att / cnt_overall]
        prnt = 'Sample {}/{}, IoU_acc:{:.2f}, Hit_acc:{:.2f}, Att_acc:{:.2f} \r'.format(var[0], var[1], var[2], var[3], var[4])
        sys.stdout.write(prnt)
        sys.stdout.flush()

    hit_acc = cnt_correct_hit / cnt_overall
    iou_acc = cnt_correct / cnt_overall
    att_acc = cnt_correct_att / cnt_overall
    return iou_acc, hit_acc, att_acc

def calc_correctness(annot, heatmap, orig_img_shape):
    bbox_dict = heat2bbox(heatmap,orig_img_shape)
    bbox, bbox_norm, bbox_score = filter_bbox(bbox_dict=bbox_dict, order='xyxy')
    bbox_norm_annot = union(annot['bbox_norm'])
    bbox_annot = union(annot['bbox'])
    bbox_norm_pred = union(bbox_norm)
    bbox_correctness = isCorrect(bbox_norm_annot, bbox_norm_pred, iou_thr=.5)
    hit_correctness = isCorrectHit(bbox_annot, heatmap, orig_img_shape)
    att_correctness = attCorrectNess(bbox_annot,heatmap,orig_img_shape)
    return bbox_correctness, hit_correctness, att_correctness

def attCorrectNess(bbox_annot,heatmap,orig_img_shape):
    H,W = orig_img_shape
    heatmap_resized = cv2.resize(heatmap, (W, H))
    h_s = np.sum(heatmap_resized)
    if h_s==0:
        return 0
    else:
        heatmap_resized /= h_s
    att_correctness = 0
    for bbox in bbox_annot:
        x0,y0,x1,y1=bbox
        att_correctness+=np.sum(heatmap_resized[int(y0):int(y1),int(x0):int(x1)])
    return att_correctness


def union(bbox):
    '''
    find how much common
    '''
    if len(bbox)==0:
        return []
    if type(bbox[0]) == type(0.0) or type(bbox[0]) == type(0):
        bbox = [bbox]
    maxes = np.max(bbox,axis=0)
    mins = np.min(bbox,axis=0)
    return [[mins[0],mins[1],maxes[2],maxes[3]]]

def isCorrect(bbox_annot, bbox_pred, iou_thr=.4):
    '''
    '''
    for bbox_p in bbox_pred:
        for bbox_a in bbox_annot:
            if IoU(bbox_p,bbox_a)>=iou_thr:
                return 1
    return 0


def isCorrectHit(bbox_annot,heatmap,orig_img_shape):
    '''
    Finds if hit accuracy acheive
    '''
    H, W = orig_img_shape
    heatmap_resized = cv2.resize(heatmap, (W, H))
    max_loc = np.unravel_index(np.argmax(heatmap_resized, axis=None), heatmap_resized.shape)
    for bbox in bbox_annot:
        if bbox[0] <= max_loc[1] <= bbox[2] and bbox[1] <= max_loc[0] <= bbox[3]:
            return 1
    return 0


def generate_report():
    pdf = PdfPages("VGG_Depth_VG_on_Flickr30k_Level2.pdf")
    for k, doc_id in enumerate(dict_test):
        prnt = 'Sample {}/{} \r'.format(k, len(dict_test))
        sys.stdout.write(prnt)
        sys.stdout.flush()
        img = np.reshape(cv2.resize(dict_test[doc_id]['img'], (299, 299)), (1, 299, 299, 3))
        orig_img_shape = dict_test[doc_id]['size']
        sen_batch = list(dict_test[doc_id]['sentences'].keys())
        img_batch = np.repeat(img, len(sen_batch), axis=0)
        tensor_list = [heatmap_w, heatmap_s, R_i, R_s, lvl_idx_wrd, lvl_idx_sen]
        feed_dict = {'input_img:0': img_batch, "text_input:0": sen_batch, "mode:0": 'test'}
        # feed_dict = {"input_img": img_batch, text_batch: sen_batch, mode: 'test'}
        qry_heats, sen_heat, qry_scores, sen_score, w_idx, sen_idx = sess.run(tensor_list, feed_dict)
        for c, sen in enumerate(sen_batch):
            if c > 0:
                continue
            title_str = sen  # +',  Sen-Img Score: %.2f'%sen_score
            title = "\n".join(wrap(title_str, 120))
            heatmap = sen_heat[c]
            fig = img_heat_bbox_disp(img[0, :], heatmap, title=title, en_name=None, bboxes=[], order='xyxy', show=False,
                                     dot_max=True)
            pdf.savefig(fig)
            for query in dict_test[doc_id]['sentences'][sen]:
                # reject not groundable/acceptable queries
                idx = dict_test[doc_id]['sentences'][sen][query]['idx']
                if len(query.split()) == 0 or len(idx) == 0:
                    continue
                annot = dict_test[doc_id]['sentences'][sen][query]
                category = annot['category']
                if 'notvisual' in category or len(annot['bbox_norm']) == 0:
                    continue
                if not check_percent(union(annot['bbox_norm'])):
                    continue
                # if reaches this point, it is groundable/acceptable

                if np.mean(qry_scores[c, idx]) == 0:
                    continue
                else:
                    heatmap = np.average(qry_heats[c, idx, :], weights=qry_scores[c, idx], axis=0)
                    strs_idx = [str(s) for s in w_idx[c, idx]]
                    en_txt = query + ', %.2f' % np.mean(qry_scores[c, idx]) + ', ' + ','.join(strs_idx)
                    fig = img_heat_bbox_disp(img[0, :], heatmap, title=title, en_name=en_txt, bboxes=annot['bbox_norm'],
                                             order='xyxy', show=False, dot_max=True)
                    pdf.savefig(fig)
    #     if k==500:
    #         break
    pdf.close()
    pdf.save()


# bbox generation config
rel_peak_thr = .3
rel_rel_thr = .3
ioa_thr = .6
topk_boxes = 3


def heat2bbox(heat_map, original_image_shape):
    '''
    code to convert heatmap to box
    '''

    h, w = heat_map.shape
    bounding_boxes = []

    heat_map = heat_map - np.min(heat_map)
    heat_map = heat_map / np.max(heat_map)

    bboxes = []
    box_scores = []

    peak_coords = peak_local_max(heat_map, exclude_border=False,
                                 threshold_rel=rel_peak_thr)  # find local peaks of heat map

    heat_resized = cv2.resize(heat_map, (
    original_image_shape[1], original_image_shape[0]))  ## resize heat map to original image shape
    peak_coords_resized = ((peak_coords + 0.5) *
                           np.asarray([original_image_shape]) /
                           np.asarray([[h, w]])
                           ).astype('int32')

    for pk_coord in peak_coords_resized:
        pk_value = heat_resized[tuple(pk_coord)]
        mask = heat_resized > pk_value * rel_rel_thr
        labeled, n = ndi.label(mask)
        l = labeled[tuple(pk_coord)]
        yy, xx = np.where(labeled == l)
        min_x = np.min(xx)
        min_y = np.min(yy)
        max_x = np.max(xx)
        max_y = np.max(yy)
        bboxes.append((min_x, min_y, max_x, max_y))
        box_scores.append(pk_value)  # you can change to pk_value * probability of sentence matching image or etc.

    ## Merging boxes that overlap too much
    box_idx = np.argsort(-np.asarray(box_scores))
    box_idx = box_idx[:min(topk_boxes, len(box_scores))]
    bboxes = [bboxes[i] for i in box_idx]
    box_scores = [box_scores[i] for i in box_idx]

    to_remove = []
    for iii in range(len(bboxes)):
        for iiii in range(iii):
            if iiii in to_remove:
                continue
            b1 = bboxes[iii]
            b2 = bboxes[iiii]
            isec = max(min(b1[2], b2[2]) - max(b1[0], b2[0]), 0) * max(min(b1[3], b2[3]) - max(b1[1], b2[1]), 0)
            ioa1 = isec / ((b1[2] - b1[0]) * (b1[3] - b1[1]))
            ioa2 = isec / ((b2[2] - b2[0]) * (b2[3] - b2[1]))
            if ioa1 > ioa_thr and ioa1 == ioa2:
                to_remove.append(iii)
            elif ioa1 > ioa_thr and ioa1 >= ioa2:
                to_remove.append(iii)
            elif ioa2 > ioa_thr and ioa2 >= ioa1:
                to_remove.append(iiii)

    for i in range(len(bboxes)):
        if i not in to_remove:
            bounding_boxes.append({
                'score': box_scores[i],
                'bbox': bboxes[i],
                'bbox_normalized': np.asarray([
                    bboxes[i][0] / heat_resized.shape[1],
                    bboxes[i][1] / heat_resized.shape[0],
                    bboxes[i][2] / heat_resized.shape[1],
                    bboxes[i][3] / heat_resized.shape[0],
                ]),
            })

    return bounding_boxes


def check_percent(bboxes):
    for bbox in bboxes:
        x_length = bbox[2] - bbox[0]
        y_length = bbox[3] - bbox[1]
        if x_length * y_length < .05:
            return False
    return True


def img_heat_bbox_disp(image, heat_map, title='', en_name='', alpha=0.6, cmap='viridis', cbar='False', dot_max=False,
                       bboxes=[], order=None, show=True):
    thr_hit = 1  # a bbox is acceptable if hit point is in middle 85% of bbox area
    thr_fit = .60  # the biggest acceptable bbox should not exceed 60% of the image
    H, W = image.shape[0:2]
    # resize heat map
    heat_map_resized = cv2.resize(heat_map, (H, W))

    # display
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(title, size=15)
    ax = plt.subplot(1, 3, 1)
    plt.imshow(image)
    if dot_max:
        max_loc = np.unravel_index(np.argmax(heat_map_resized, axis=None), heat_map_resized.shape)
        plt.scatter(x=max_loc[1], y=max_loc[0], edgecolor='w', linewidth=3)

    if len(bboxes) > 0:  # it gets normalized bbox
        if order == None:
            order = 'xxyy'

        for i in range(len(bboxes)):
            bbox_norm = bboxes[i]
            if order == 'xxyy':
                x_min, x_max, y_min, y_max = int(bbox_norm[0] * W), int(bbox_norm[1] * W), int(bbox_norm[2] * H), int(
                    bbox_norm[3] * H)
            elif order == 'xyxy':
                x_min, x_max, y_min, y_max = int(bbox_norm[0] * W), int(bbox_norm[2] * W), int(bbox_norm[1] * H), int(
                    bbox_norm[3] * H)
            x_length, y_length = x_max - x_min, y_max - y_min
            box = plt.Rectangle((x_min, y_min), x_length, y_length, edgecolor='w', linewidth=3, fill=False)
            plt.gca().add_patch(box)
            if en_name != '':
                ax.text(x_min + .5 * x_length, y_min + 10, en_name,
                        verticalalignment='center', horizontalalignment='center',
                        # transform=ax.transAxes,
                        color='white', fontsize=15)
                # an = ax.annotate(en_name, xy=(x_min,y_min), xycoords="data", va="center", ha="center", bbox=dict(boxstyle="round", fc="w"))
                # plt.gca().add_patch(an)

    plt.imshow(heat_map_resized, alpha=alpha, cmap=cmap)

    # plt.figure(2, figsize=(6, 6))
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    # plt.figure(3, figsize=(6, 6))
    plt.subplot(1, 3, 3)
    plt.imshow(heat_map_resized)
    fig.tight_layout()
    fig.subplots_adjust(top=.85)

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def crop_resize_im(image, bbox, size, order='xxyy'):
    H, W, _ = image.shape
    if order == 'xxyy':
        roi = image[int(bbox[2] * H):int(bbox[3] * H), int(bbox[0] * W):int(bbox[1] * W), :]
    elif order == 'xyxy':
        roi = image[int(bbox[1] * H):int(bbox[3] * H), int(bbox[0] * W):int(bbox[2] * W), :]
    roi = cv2.resize(roi, size)
    return roi


def im2double(im):
    return cv2.normalize(im.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

def filter_bbox(bbox_dict,order=None):
    thr_fit = .90 #the biggest acceptable bbox should not exceed 80% of the image
    if order==None:
            order='xxyy'
        
    filtered_bbox = []
    filtered_bbox_norm = []
    filtered_score = []
    if len(bbox_dict)>0: #it gets normalized bbox
        for i in range(len(bbox_dict)):
            bbox = bbox_dict[i]['bbox']
            bbox_norm = bbox_dict[i]['bbox_normalized']
            bbox_score = bbox_dict[i]['score']
            if order=='xxyy':
                x_min,x_max,y_min,y_max = bbox_norm[0],bbox_norm[1],bbox_norm[2],bbox_norm[3]
            elif order=='xyxy':
                x_min,x_max,y_min,y_max = bbox_norm[0],bbox_norm[2],bbox_norm[1],bbox_norm[3]
            if bbox_score>0:
                x_length,y_length = x_max-x_min,y_max-y_min
                if x_length*y_length<thr_fit:
                    filtered_score.append(bbox_score)
                    filtered_bbox.append(bbox)
                    filtered_bbox_norm.append(bbox_norm)
    return filtered_bbox, filtered_bbox_norm, filtered_score



def IoU(boxA, boxB):
    # order = xyxy
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
