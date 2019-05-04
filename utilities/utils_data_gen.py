import tensorflow as tf
import random
from utils_model import annToMask
import pycocotools.mask as maskUtils
import numpy as np
import cv2

# Print Tensors
def tf_print(tensor, transform=None):
    # Insert a custom python operation into the graph that does nothing but print a tensors value
    def print_tensor(x):
        # x is typically a numpy array here so you could do anything you want with it,
        # but adding a transformation of some kind usually makes the output more digestible
        print(x if transform is None else transform(x))
        return x

    log_op = tf.py_func(print_tensor, [tensor], [tensor.dtype])[0]
    with tf.control_dependencies([log_op]):
        res = tf.identity(tensor)

    # Return the given tensor
    return res

def print_tensors(graph):
    for op in graph.get_operations():
        print(op.values())
        


# Laod images in Memory
def preload_images(all_ids_list, db):
    valid_ids = []
    images_dict = {}
    for i, id_train in enumerate(all_ids_list):
        imgbin = db.get(id_train.encode('utf-8'))
        if imgbin == None:
            continue
        buff = np.frombuffer(imgbin, dtype='uint8')
        imgbgr = cv2.imdecode(buff, cv2.IMREAD_COLOR)
        img = imgbgr[:, :, [2, 1, 0]]
        im = cv2.resize(img, (299, 299))
        valid_ids.append(id_train)
        images_dict[id_train] = im
        if i % 10000 == 0:
            print('Processed: ', i, ' images')
    print('Finished preprocessing images.')
    return valid_ids, images_dict

def batch_gen(valid_ids, annot_dict, train_images, n_batch):
    img_batch = np.empty((n_batch, 299, 299, 3), dtype='float32')
    cap_batch = []

    chosen_ids = random.sample(valid_ids, n_batch)
    for i, chosen_id in enumerate(chosen_ids):
        imgbin = train_images[chosen_id]
        img_batch[i,:,:,:] = imgbin
        queries = [annot['query'] for annot in annot_dict[chosen_id]['annotations']]
        sentence = random.choice(queries)
        cap_batch.append(sentence)
        
    return img_batch, cap_batch

# Batch Gen Seg
def batch_gen_seg(valid_ids, annot_dict, train_images, n_batch, msk_size):
    img_batch = np.empty((n_batch, 299, 299, 3), dtype='float32')
    cap_batch = []
    cat_batch = []
    msk_batch = np.empty((n_batch, 100, msk_size, msk_size))  # maximum 100 category per-image

    # box_batch = np.zeros((n_batch,100,4),dtype='float32')
    # currently, it takes negative samples randomly from "all" dataset
    # it randomly picks any batch, so it doesn't have ending
    chosen_ids = random.sample(valid_ids, n_batch)
    for i, chosen_id in enumerate(chosen_ids):
        imgbin = train_images[chosen_id]
        img_batch[i, :, :, :] = imgbin
        sentence = random.choice(annot_dict[chosen_id]['sentences'])
        cap_batch.append(sentence)
        size = annot_dict[chosen_id]['size']
        cat_sen = []
        # box_list = []
        for j, ann in enumerate(annot_dict[chosen_id]['annotations']):
            cat = '_'.join(ann['category'].split())
            cat_sen.append(cat)
            msk_batch[i, j, :, :] = cv2.resize(annToMask(ann, size), (msk_size, msk_size))
            # box_batch[i,j,:] = ann['bbox_norm']
        cat_batch.append(' '.join(cat_sen))
    return img_batch, cap_batch, cat_batch, msk_batch


def batch_gen_seg_add_cat_map(valid_ids, annot_dict, train_images, n_batch,msk_size):
    #img_batch = np.empty((n_batch, 299, 299, 3), dtype='float32')
    #cap_batch = []
    cat_batch = []
    msk_batch = np.empty((n_batch, 100, msk_size, msk_size))  # maximum 100 category per-image

    chosen_ids = random.sample(valid_ids, n_batch)
    for i, chosen_id in enumerate(chosen_ids):
        #imgbin = train_images[chosen_id]
        #img_batch[i, :, :, :] = imgbin
        #sentence = random.choice(annot_dict[chosen_id]['sentences'])
        #cap_batch.append(sentence)
        size = annot_dict[chosen_id]['size']
        cat_sen = []
        # box_list = []
        cat_dict = {}
        for j, ann in enumerate(annot_dict[chosen_id]['annotations']):
            cat = '_'.join(ann['category'].split())
            cat_mask = annToMask(ann, size)
            if cat not in cat_dict:
                cat_dict[cat] = cat_mask
            else:
                cat_dict[cat] = cat_dict[cat] + cat_mask
        counter = 0
        for k, v in cat_dict.items():
            cat_sen.append(k)
            np.clip(v, 0, 1, out=v)
            msk_batch[i, counter, :, :] = cv2.resize(v, (msk_size, msk_size))
            counter += 1
        cat_batch.append(' '.join(cat_sen))

    return cat_batch, msk_batch

def batch_gen_old(ids, annot_dict, txn):
    
    img_batch = np.empty((n_batch, 299, 299, 3), dtype='float32')
    cap_batch = []
   #currently, it takes negative samples randomly from “all” dataset
   #it randomly picks any batch, so it doesn’t have ending
    seen = {}
    for i in range(n_batch):
        choice_id = random.choice(ids)
        while choice_id in seen: #we don’t want to have repetitive img/caps in a batch
            choice_id = random.choice(ids)
        imgbin = txn.get(choice_id.encode('utf-8'))
        if imgbin!=None:
            buff = np.frombuffer(imgbin, dtype='uint8')
        else:
            buff = []
        while choice_id in seen or len(buff)==0:
            choice_id = random.choice(ids)
            imgbin = txn.get(choice_id.encode('utf-8'))
            if imgbin!=None:
                buff = np.frombuffer(imgbin, dtype='uint8')
            else:
                buff = []
        seen[choice_id] = 1

        imgbgr = cv2.imdecode(buff, cv2.IMREAD_COLOR)
        img = imgbgr[:,:,[2,1,0]]
        img_batch[i,:,:,:] = cv2.resize(img,(299,299))
        queries = [annot['query'] for annot in dict_train[choice_id]['annotations']]
        sentence = random.choice(queries)
        cap_batch.append(sentence)
    return img_batch, cap_batch
