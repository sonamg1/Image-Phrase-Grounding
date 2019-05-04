import json
import os
import tensorflow as tf
from skimage.feature import peak_local_max
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import tensorflow.contrib.slim as slim
import sys
#slim_models_path = '/home/david_hc95/grounding/iccv19_grounding/models/'
slim_models_path = '/home/sonam/models/'
sys.path.append(slim_models_path)
import pycocotools.mask as maskUtils

# Load Pretrain Model
def load_model(model_path,config):
    new_graph = tf.Graph()
    sess = tf.InteractiveSession(graph = new_graph, config=config)
    new_saver = tf.train.import_meta_graph(model_path+'.meta')
    _ = sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    new_saver.restore(sess, model_path)
    return sess, new_graph


# Ann to Mask
def annToMask(ann,size):
    h,w,_ = size
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    m = maskUtils.decode(rle)
    return m

# Preprocess Images
def pre_process(image, pre_processing_name):
    if pre_processing_name == 'vgg_preprocessing':
        with tf.variable_scope(pre_processing_name):
            image = tf.subtract(image, [123.68, 116.78, 103.94])

    elif pre_processing_name == 'inception_preprocessing':
        with tf.variable_scope(pre_processing_name):
            # image should be in range [-1,1]
            image = tf.divide(image, 255.0)
            image = tf.subtract(image, .5)
            image = tf.multiply(image, 2.0)
    else:
        raise ValueError('Provide a valid/supported model name.')
        return
    return image

def add_1by1_conv(feat_map, n_layers, n_filters, name, regularizer):
    with tf.variable_scope(name + '_postConv'):
        for i in range(n_layers):
            with tf.variable_scope(name + '_stage_' + str(i)):
                feat_map = tf.layers.conv2d(feat_map, filters=n_filters[i], kernel_size=[1, 1],
                                            kernel_regularizer=regularizer)
                feat_map = tf.nn.leaky_relu(feat_map, alpha=.25)
    return feat_map


def add_3by3_conv(feat_map, n_layers, n_filters, name, regularizer):
    with tf.variable_scope(name + '_postConv'):
        for i in range(n_layers):
            with tf.variable_scope(name + '_stage_' + str(i)):
                feat_map = tf.layers.conv2d(feat_map, filters=n_filters[i], kernel_size=[3, 3],
                                            kernel_regularizer=regularizer, padding='same')
                feat_map = tf.nn.leaky_relu(feat_map, alpha=.25)
    return feat_map


def depth_selection_3layer_vg(model, regularizer, conv_method=3, n_layers=1):
    # vgg conv5_1, conv5_3
    # vgg conv5_1, conv5_3 , size is 37x37
    if conv_method == 3:
        conv = add_3by3_conv
    else:
        conv = add_1by1_conv

    with tf.variable_scope('stack_v'):
        v1 = tf.identity(model['vgg_16/conv5/conv5_1'], name='v1')
        v1 = conv(v1, n_layers, n_filters=[1024, 1024, 1024], name='v1', regularizer=regularizer)
        size = v1.get_shape().as_list()[1:3]
        resize_method = tf.image.ResizeMethod.BILINEAR
        v2 = tf.identity(model['vgg_16/conv5/conv5_3'], name='v2')
        v2 = tf.image.resize_images(v2, size, method=resize_method)
        v2 = conv(v2, n_layers, n_filters=[1024, 1024, 1024], name='v2', regularizer=regularizer)
        v3 = tf.identity(model['vgg_16/conv4/conv4_1'], name='v3')
        v3 = tf.image.resize_images(v3, size, method=resize_method)
        v3 = conv(v3, n_layers, n_filters=[1024, 1024, 1024], name='v3', regularizer=regularizer)
        v_all = tf.stack([v1, v2, v3], axis=3)
        v_all = tf.reshape(v_all, [-1, v_all.shape[1] * v_all.shape[2], v_all.shape[3], v_all.shape[4]])
        v_all = tf.nn.l2_normalize(v_all, axis=-1, name='stacked_image_feature_maps')  # ?

    return v_all


def depth_selection_vgg(model, regularizer, conv_kernel_size=3, n_layers=1):
    '''
    downsample first 2 layers to 18x18 concat and pass through 3x3 conv 
    '''
    if conv_kernel_size == 3:
        conv_method = add_3by3_conv
    else:
        conv_method = add_1by1_conv

    with tf.variable_scope('stack_v'):
        v1 = tf.identity(model['vgg_16/conv5/conv5_1'], name='v1') # 37×37×512
        v1 = conv_method (v1, n_layers, n_filters=[1024, 1024, 1024], name='v1', regularizer=regularizer)
        size = v1.get_shape().as_list()[1:3]
        resize_method = tf.image.ResizeMethod.BILINEAR
        v2 = tf.identity(model['vgg_16/conv5/conv5_3'], name='v2') # 37×37×512
        # v2 = tf.image.resize_images(v2, size, method=resize_method)
        v2 = conv_method (v2, n_layers, n_filters=[1024, 1024, 1024], name='v2', regularizer=regularizer)
        v3 = tf.identity(model['vgg_16/conv4/conv4_1'], name='v3') # 18×18×512
        v3 = tf.image.resize_images(v3, size, method=resize_method)
        v3 = conv_method (v3, n_layers, n_filters=[1024, 1024, 1024], name='v3', regularizer=regularizer)
        v4 = tf.identity(model['vgg_16/conv4/conv4_3'], name='v4') # 18×18×512
        v4 = tf.image.resize_images(v4, size, method=resize_method)
        v4 = conv_method (v4, n_layers, n_filters=[1024, 1024, 1024], name='v4', regularizer=regularizer)
        v_all = tf.stack([v1, v2, v3, v4], axis=3)
        v_all = tf.reshape(v_all, [-1, v_all.shape[1] * v_all.shape[2], v_all.shape[3], v_all.shape[4]])
        v_all = tf.nn.l2_normalize(v_all, axis=-1, name='stacked_image_feature_maps')
    return v_all


def depth_selection_pnasnet(model, regularizer, conv_kernel_size=3, n_layers = 3):
    '''
   upsample last 2 layers to 19x19 change to 1024 channels
    
    '''
    conv_method = None
    if conv_kernel_size == 1:
        conv_method = add_1by1_conv
    elif conv_kernel_size == 3:
        conv_method = add_3by3_conv
    else:
        raise ValueError('Invalid conv_kernel_size parameter. Should be either 1 or 3.')
    with tf.variable_scope('stack_v'):
        v1 = tf.identity(model['Cell_5'],name='v1')    # 19×19×2160
        v1 = conv_method(v1,n_layers,n_filters=[1024],name='v1',regularizer=regularizer)  
        size = v1.get_shape().as_list()[1:3]
        resize_method = tf.image.ResizeMethod.BILINEAR
        v2 = tf.identity(model['Cell_7'],name='v2')    # 19×19×2160
       #v2 = tf.image.resize_images(v2, size, method=resize_method)
        v2 = conv_method(v2,n_layers,n_filters=[1024],name='v2',regularizer=regularizer) 
        v3 = tf.identity(model['Cell_9'],name='v3')              # 10×10×4320
        v3 = tf.image.resize_images(v3, size, method=resize_method)
        v3 = conv_method(v3,n_layers,n_filters=[1024],name='v3',regularizer=regularizer)
        v4 = tf.identity(model['Cell_11'],name='v4')   # 10×10×4320
        v4 = tf.image.resize_images(v4, size, method=resize_method)
        v4 = conv_method(v4,n_layers,n_filters=[1024],name='v4',regularizer=regularizer)
        v_all = tf.stack([v1,v2,v3,v4], axis=3)
        v_all = tf.reshape(v_all,[-1,v_all.shape[1]*v_all.shape[2],v_all.shape[3],v_all.shape[4]])
        v_all = tf.nn.l2_normalize(v_all, axis=-1, name='stacked_image_feature_maps')
    return v_all

def depth_selection_pyramid_vgg(model,regularizer, conv_kernel_size = 3, n_layers=1):
    '''
    take 2 layers upsample to 37x37 and then concatenate all 4 and pass thorugh 
    3x3 conv to reduce no of channels to 1024
    '''
    conv_method = None
    if conv_kernel_size == 1:
        conv_method = add_1by1_conv
    elif conv_kernel_size == 3:
        conv_method = add_3by3_conv
    else:
        raise ValueError('Invalid conv_kernel_size parameter. Should be either 1 or 3.')

    with tf.variable_scope('stack_v'):
        v4 = tf.identity(model['vgg_16/conv4/conv4_3'], name='v4')
        v3 = tf.identity(model['vgg_16/conv4/conv4_1'], name='v3')
        v2 = tf.identity(model['vgg_16/conv5/conv5_3'], name='v2')
        v1 = tf.identity(model['vgg_16/conv5/conv5_1'], name='v1')
        v2 = tf.layers.conv2d_transpose(v2, filters=512, kernel_size=(3, 3), strides=(2, 2))
        v1 = tf.layers.conv2d_transpose(v1, filters=512, kernel_size=(3, 3), strides=(2, 2))
        v_all = tf.concat([v4, v3, v2, v1], axis=3)
        v_all = conv_method(v_all, n_layers, n_filters=[1024, 1024, 1024], name='v_all', regularizer=regularizer)
        # flattening image
        v_all = tf.reshape(v_all, [-1, v_all.shape[1] * v_all.shape[2], v_all.shape[3]])
        v_all = tf.nn.l2_normalize(v_all, axis=-1, name='stacked_image_feature_maps')
        v_all = tf.expand_dims(v_all, 2)  # ?XNX1XD
    return v_all

def depth_selection_pyramid_pnas(model, regularizer, conv_kernel_size =3, n_layers = 1):
    '''
    vis_model.outputs
    take last layer then resize to second layer and then 3x3 conv
    second, third layer 3x3 conv to 38x38 and then 1x1 conv on first layer to bring to same space
    then concat all and pass 3x3 conv to combine
    '''
    conv_method = None
    if conv_kernel_size == 1:
        conv_method = add_1by1_conv
    elif conv_kernel_size == 3:
        conv_method = add_3by3_conv
    else:
        raise ValueError('Invalid conv_kernel_size parameter. Should be either 1 or 3.')
   
    with tf.variable_scope('stack_v'): 
        v1 = tf.identity(model['Cell_3'],name='v1') #38x38x1080
        v2 = tf.identity(model['Cell_5'],name='v2') #19x19x2160
        v3 = tf.identity(model['Cell_7'],name='v3') #19x19x2160
        v4 = tf.identity(model['Cell_9'],name='v4')#10x10x4320
        
        resize_method = tf.image.ResizeMethod.BILINEAR
        size = v2.get_shape().as_list()[1:3]
        v4 = tf.image.resize_images(v4,size , method=resize_method)#19x19x4320
        v4 = tf.layers.conv2d_transpose(v4, filters = 512, kernel_size=(2,2), strides = (2,2), padding=
                                       'valid')
        v3 = tf.layers.conv2d_transpose(v3, filters = 512, kernel_size=(2,2), strides = (2,2), padding=
                                       'valid')
        v2 = tf.layers.conv2d_transpose(v2, filters = 512, kernel_size=(2,2), strides = (2,2), padding=
                                       'valid')
        v1 = tf.layers.conv2d(v1, filters=512, kernel_size=[1,1], padding='same')
        
        v_all  = tf.concat([v4,v3,v2,v1], axis = 3)
        v_all = conv_method(v_all,n_layers=1,n_filters=[1024],name='v_all',regularizer=regularizer)
        v_all = tf.reshape(v_all,[-1,v_all.shape[1]*v_all.shape[2],v_all.shape[3]])
        v_all = tf.nn.l2_normalize(v_all, axis=-1, name='stacked_image_feature_maps')
        v_all = tf.expand_dims(v_all, 2)
    
    return v_all

def depth_selection_newattn_vgg(model, regularizer, conv_kernel_size =3, n_layers=1):
    
    conv_method = None
    if conv_kernel_size == 1:
        conv_method = add_1by1_conv
    elif conv_kernel_size == 3:
        conv_method = add_3by3_conv
    else:
        raise ValueError('Invalid conv_kernel_size parameter. Should be either 1 or 3.')

    with tf.variable_scope('stack_v'):
        v4 = tf.identity(model['vgg_16/conv4/conv4_3'], name='v4')
        v3 = tf.identity(model['vgg_16/conv4/conv4_1'], name='v3')
        v2 = tf.identity(model['vgg_16/conv5/conv5_3'], name='v2')
        v1 = tf.identity(model['vgg_16/conv5/conv5_1'], name='v1')
        
        v4 = tf.layers.conv2d(v4, 512, (2,2), (2,2))
        v3 = tf.layers.conv2d(v3, 512, (2,2), (2,2))
        v_all = tf.concat([v4, v3, v2, v1], axis=3) # 18x18x2048
        v_all = conv_method(v_all, n_layers, n_filters=[512,512,512], name='v_all',regularizer=regularizer)
        
        # flattening image
        v_all = tf.reshape(v_all, [-1, v_all.shape[1] * v_all.shape[2], v_all.shape[3]])
        v_all = tf.nn.l2_normalize(v_all, axis=-1, name='stacked_image_feature_maps')
        v_all = tf.expand_dims(v_all, 2)  # ?XNX1XD
        
    return v_all




# Load Pretrained Model
class pre_trained_load():
    """
    Building a TF graph based on pre-trained image classification models.
    This implementation supports Slim.
    """

    def __init__(self, model_name, image_shape=(None, 224, 224, 3),
                 input_tensor=None, session=None, is_training=True,
                 global_pool=False, num_classes=None):
        supported_list = ['vgg_16', 'InceptionV3', 'InceptionV4', 'pnasnet_large', 'resnet_v2_152', 'resnet_v2_101',
                          'resnet_v2_50']
        if model_name not in supported_list:
            raise ValueError('Provide a valid/supported model name.')
            return
        self.model_name = model_name
        self.image_shape = image_shape
        self.is_training = is_training
        self.global_pool = global_pool
        self.num_classes = num_classes
        self._build_graph(input_tensor)
        self.sess = session

    def _build_graph(self, input_tensor):
        with tf.name_scope('inputs'):
            if input_tensor is None:
                input_tensor = tf.placeholder(tf.float32, shape=self.image_shape, name='input_img')
            else:
                assert self.image_shape == tuple(input_tensor.shape.as_list())
            self.input_tensor = input_tensor

        if self.model_name == 'vgg_16':
            self.ckpt_path = slim_models_path + "vgg_16.ckpt"
            from nets.vgg import vgg_16, vgg_arg_scope
            with slim.arg_scope(vgg_arg_scope()):
                self.output, self.outputs = vgg_16(self.input_tensor, num_classes=self.num_classes,
                                                   is_training=self.is_training, global_pool=self.global_pool)

        if self.model_name == 'resnet_v2_152':
            self.ckpt_path = slim_models_path + "resnet_v2_152.ckpt"
            from nets.resnet_v2 import resnet_v2_152, resnet_arg_scope
            with slim.arg_scope(resnet_arg_scope()):
                self.output, self.outputs = resnet_v2_152(self.input_tensor, num_classes=self.num_classes,
                                                          is_training=self.is_training, global_pool=self.global_pool)

        elif self.model_name == 'resnet_v2_101':
            self.ckpt_path = slim_models_path + "resnet_v2_101.ckpt"
            from nets.resnet_v2 import resnet_v2_101, resnet_arg_scope
            with slim.arg_scope(resnet_arg_scope()):
                self.output, self.outputs = resnet_v2_101(self.input_tensor, num_classes=self.num_classes,
                                                          is_training=self.is_training, global_pool=self.global_pool)

        elif self.model_name == 'resnet_v2_50':
            self.ckpt_path = slim_models_path + "resnet_v2_50.ckpt"
            from nets.resnet_v2 import resnet_v2_50, resnet_arg_scope
            with slim.arg_scope(resnet_arg_scope()):
                self.output, self.outputs = resnet_v2_50(self.input_tensor, num_classes=self.num_classes,
                                                         is_training=self.is_training, global_pool=self.global_pool)

        elif self.model_name == 'InceptionV3':
            self.ckpt_path = slim_models_path + "inception_v3.ckpt"
            from nets.inception import inception_v3, inception_v3_arg_scope
            with slim.arg_scope(inception_v3_arg_scope()):
                self.output, self.outputs = inception_v3(self.input_tensor, num_classes=self.num_classes,
                                                         is_training=self.is_training)

        elif self.model_name == 'InceptionV4':
            self.ckpt_path = slim_models_path + "inception_v4.ckpt"
            from nets.inception import inception_v4, inception_v4_arg_scope
            with slim.arg_scope(inception_v4_arg_scope()):
                self.output, self.outputs = inception_v4(self.input_tensor, num_classes=self.num_classes,
                                                         is_training=self.is_training)

        elif self.model_name == 'pnasnet_large':
            self.ckpt_path = slim_models_path + "pnasnet_large_2.ckpt"
            from nets.nasnet.pnasnet import build_pnasnet_large, pnasnet_large_arg_scope
            with tf.variable_scope(self.model_name):
                with slim.arg_scope(pnasnet_large_arg_scope()):
                    self.output, self.outputs = build_pnasnet_large(self.input_tensor, num_classes=self.num_classes,
                                                                    is_training=self.is_training)

        # collecting all variables related to this model
        # self.model_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.model_name+'/model')
        self.model_weights = slim.get_model_variables(self.model_name)

    def load_weights(self):
        model_saver = tf.train.Saver(self.model_weights)
        model_saver.restore(self.sess, self.ckpt_path)

    def __getitem__(self, key):
        return self.outputs[key]


def attn(e_w, v, e_s):
    ## Inputs: local and global cap and img features ##
    ## Output: Heatmap for each word, Global Heatmap, Attnded Vis features, Corr-vals
    # e: ?xTxD, v: ?xNx4xD, e_bar: ?xD

    with tf.variable_scope('attention'):
        ###word-level###
        # heatmap pool
        h = tf.nn.relu(tf.einsum('bij,bklj->bikl', e_w, v))  # pair-wise ev^T: ?xTxNx4
        # attention
        a = tf.einsum('bijk,bjkl->bilk', h, v)  # ?xTxDx4 attnded visual reps for each of T words
        # pair-wise score
        a_norm = tf.nn.l2_normalize(a, axis=2)
        e_w_norm = tf.nn.l2_normalize(e_w, axis=2)
        R_ik = tf.einsum('bilk,bil->bik', a_norm, e_w_norm)  # cosine for T (words,img_reps) for all pairs
        R_ik = tf.identity(R_ik, name='level_score_word')
        R_i = tf.reduce_max(R_ik, axis=-1, name='score_word')  # ?xT
        # R = tf.log(tf.pow(tf.reduce_sum(tf.exp(gamma_1*R_i),axis=1),1/gamma_1)) #? corrs
        # heatmap
        idx_i = tf.argmax(R_ik, axis=-1, name='level_index_word')  # ?xT index of the featuremap which maximizes R_i
        with tf.name_scope('summaries'):
            tf.summary.histogram('histogram_w', idx_i)
        ii, jj = tf.meshgrid(tf.range(tf.shape(idx_i)[0]), tf.range(tf.shape(idx_i)[1]), indexing='ij')
        ii = tf.cast(ii, tf.int64)
        jj = tf.cast(jj, tf.int64)
        batch_idx_i = tf.stack([tf.reshape(ii, (-1,)),
                                tf.reshape(jj, (-1,)),
                                tf.reshape(idx_i, (-1,))], axis=1)  # ?Tx3 indices of argmax
        N0 = int(np.sqrt(h.get_shape().as_list()[2]))
        h_max = tf.gather_nd(tf.transpose(h, [0, 1, 3, 2]), batch_idx_i)  # ?TxN retrieving max heatmaps

        heatmap_wd = tf.reshape(h_max, [tf.shape(h)[0], tf.shape(h)[1], N0, N0], name='heatmap_word')
        heatmap_wd_l = tf.reshape(h, [tf.shape(h)[0], tf.shape(h)[1], N0, N0, tf.shape(h)[3]],
                                  name='level_heatmap_word')

        ###sentence-level###
        # heatmap pool
        h_s = tf.nn.relu(tf.einsum('bj,blkj->blk', e_s, v))  # pair-wise e_bar*v^T: ?xNx4
        # attention
        a_s = tf.einsum('bjk,bjki->bik', h_s, v)  # ?xDx4 attnded visual reps for sen.
        # pair-wise score
        a_s_norm = tf.nn.l2_normalize(a_s, axis=1)
        e_s_norm = tf.nn.l2_normalize(e_s, axis=1)
        R_sk = tf.einsum('bik,bi->bk', a_s_norm, e_s_norm)  # cosine for (sen,img_reps)
        R_sk = tf.identity(R_sk, name='level_score_sentence')
        R_s = tf.reduce_mean(R_sk, axis=-1, name='score_sentence')  # ?
        # heatmap
        idx_k = tf.argmax(R_sk, axis=-1, name='level_index_sentence')  # ? index of the featuremap which maximizes R_i
        with tf.name_scope('summaries'):
            tf.summary.histogram('histogram_s', idx_k)
        ii_k = tf.cast(tf.range(tf.shape(idx_k)[0]), dtype='int64')
        batch_idx_k = tf.stack([ii_k, idx_k], axis=1)
        N0_g = int(np.sqrt(h_s.get_shape().as_list()[1]))
        h_s_max = tf.gather_nd(tf.transpose(h_s, [0, 2, 1]), batch_idx_k)  # ?xN retrieving max heatmaps
        heatmap_sd = tf.reshape(h_s_max, [-1, N0_g, N0_g], name='heatmap_sentence')
        heatmap_sd_l = tf.reshape(h_s, [-1, N0_g, N0_g, tf.shape(h)[3]], name='level_heatmap_sentence')
    return heatmap_wd, heatmap_sd, R_i, R_s


def attn_new(e_w, v, e_s):
    '''
    e_w:?xTXD , v:?XNx1XD, e_s:?XD
    return: heatmap_wd ?xTX (37X37) , heatmap_sd ?X(37X37)
    '''
    v = tf.squeeze(v, 2)

    with tf.variable_scope('attention'):
        resolution = v.get_shape()[1]
        dim = tf.shape(v)[2]
        wrd_len = tf.shape(e_w)[1]
        N0_g = int(np.sqrt(v.get_shape().as_list()[1]))
        ### Word-Level

        new_e_w = tf.expand_dims(e_w, 2)  # ?TX1XD
        new_e_w = tf.tile(new_e_w, [1, 1, resolution, 1])  # ?xT*NXD
        new_v = tf.expand_dims(v, 1)  # ?x1XNXD
        new_v = tf.tile(new_v, [1, wrd_len, 1, 1])  # ?xT*NXD
        v_w = tf.concat([new_e_w, new_v], axis=3)  # ?xT*NX2D

    h_w = tf.layers.dense(v_w, units=512, name='attn_space_1', reuse=tf.AUTO_REUSE)  # ?xT*NX512
    h_w = tf.math.tanh(h_w)
    h_w = tf.layers.dense(h_w, units=1, name='attn_space_2', reuse=tf.AUTO_REUSE)  # ?xT*NX1

    with tf.variable_scope('attention'):
        h_w = tf.squeeze(h_w, 3)  # ?xT*N
        h_w = tf.nn.relu(h_w)  # ?xT*N
        heatmap_wd = tf.reshape(h_w, [tf.shape(h_w)[0], tf.shape(h_w)[1], N0_g, N0_g],
                                name="heatmap_word")  # ?xTX37X37
        ### Sentence-Level ###
        new_e_s = tf.expand_dims(e_s, 1)  # ?x1XD
        new_e_s = tf.tile(new_e_s, [1, resolution, 1])  # ?xNXD
        v_s = tf.concat([new_e_s, v], axis=2)  # ?xNX2D

    h_s = tf.layers.dense(v_s, units=512, name='attn_space_1', reuse=tf.AUTO_REUSE)  # ?xNX512
    h_s = tf.math.tanh(h_s)
    h_s = tf.layers.dense(h_s, units=1, name='attn_space_2', reuse=tf.AUTO_REUSE)  # ?xNx1

    with tf.variable_scope('attention'):
        h_s = tf.squeeze(h_s, -1)
        h_s = tf.nn.relu(h_s)  # ?*N
        heatmap_sd = tf.reshape(h_s, [-1, N0_g, N0_g], name="heatmap_sentence")  # ?X37X37

    return heatmap_wd, heatmap_sd



# LOSS
def seg_loss(heatmap, mask):
    with tf.variable_scope('seg_loss'):
        # l_b = tf.nn.l2_loss(box_pred-box)
        # converting heatmap tensor(1e-7, dtype=heatmap.dtype)
        epsilon_ = tf.convert_to_tensor(1e-7, dtype=heatmap.dtype)
        heatmap_l = tf.clip_by_value(heatmap, epsilon_, 1 - epsilon_)
        heatmap_l = tf.log(heatmap_l / (1 - heatmap_l))
        # returning binary crossentropy loss
        l_m = tf.nn.sigmoid_cross_entropy_with_logits(labels=mask, logits=heatmap_l)
        l_m = tf.reduce_mean(l_m)
    return l_m


def attn_loss_new(e_w, v, e_s, gamma_1, gamma_2):
    '''
     #e: ?xTxD, v: ?xNx1xD, e_bar: ?xD
    '''
    batch_sz = tf.shape(v)[0]
    with tf.variable_scope('attention_loss'):
        ###word-level###
        # heatmap
        new_e_w_loss = tf.expand_dims(e_w, 0)  # 1x?xTxD
        new_e_w_loss = tf.tile(new_e_w_loss, [batch_sz, 1, 1, 1])  # ?x?xTxD

        new_e_s_loss = tf.expand_dims(e_s, 0)  # 1x?xD
        new_e_s_loss = tf.tile(new_e_s_loss, [batch_sz, 1, 1])  # ?x?xD

        new_v_loss = tf.expand_dims(v, 1)  # ?x1x Nx1xD
        new_v_loss = tf.tile(new_v_loss, [1, batch_sz, 1, 1, 1])  # ?x?x Nx1xD

        new_e_w_loss = tf.reshape(new_e_w_loss, [tf.shape(new_e_w_loss)[0] * tf.shape(new_e_w_loss)[1],
                                                 tf.shape(new_e_w_loss)[2], new_e_w_loss.get_shape()[3]])  # (?x?)xTxD

        new_v_loss = tf.reshape(new_v_loss, [tf.shape(new_v_loss)[0] * tf.shape(new_v_loss)[1],
                                             new_v_loss.get_shape()[2], new_v_loss.get_shape()[3],
                                             new_v_loss.get_shape()[4]])  # (?x?)x Nx1xD

        new_e_s_loss = tf.reshape(new_e_s_loss, [tf.shape(new_e_s_loss)[0] * tf.shape(new_e_s_loss)[1],
                                                 new_e_s_loss.get_shape()[2]])  # (?x?)xD

    h, h_s = attn_new(new_e_w_loss, new_v_loss, new_e_s_loss)  # hs = (?X?) *37X37 , h = (?X?)xT *37X37
    # flatten image

    with tf.variable_scope('attention_loss'):
        h = tf.reshape(h, [tf.shape(h)[0], tf.shape(h)[1], tf.shape(h)[2] * tf.shape(h)[3]])  # (?X?)xT *(37X37)
        h_s = tf.reshape(h_s, [tf.shape(h_s)[0], tf.shape(h_s)[1] * tf.shape(h_s)[2]])  # (?X?) *(37X37)

        h = tf.expand_dims(h, -1)  # (?X?)xT x(37X37)x1
        h_s = tf.expand_dims(h_s, -1)  # (?X?) *(37X37)x1

        # flattenign on batch size
        h = tf.reshape(h, [batch_sz, batch_sz, tf.shape(h)[1], tf.shape(h)[2], tf.shape(h)[3]])  # ?X?xT x(37X37)x1
        h_s = tf.reshape(h_s, [batch_sz, batch_sz, tf.shape(h_s)[1], tf.shape(h_s)[2]])  # ?X?x(37X37)x1

        # h = tf.nn.relu(tf.einsum('bij,cklj->bcikl',e_w,v)) #pair-wise ev^T: ?x?xTxNx4
        # attention
        a = tf.einsum('bcijl,cjlk->bcikl', h, v)  # ?x?xTxDx4 attnded visual reps for each of T words for all pairs
        # pair-wise score
        a_norm = tf.nn.l2_normalize(a, axis=3)
        e_w_norm = tf.nn.l2_normalize(e_w, axis=2)
        R_ik = tf.einsum('bcilk,bil->bcik', a_norm, e_w_norm)  # cosine for T (words,img_reps) for all pairs
        # level dropout
        # R_ik_sh = R_ik.get_shape().as_list()
        # R_ik = tf.layers.dropout(R_ik,rate=0.5,noise_shape=[1,1,1,R_ik_sh[3]],
        #                         training=isTraining)
        R_i = tf.reduce_max(R_ik, axis=-1)  # ?x?xT
        R = tf.log(tf.pow(tf.reduce_sum(tf.exp(gamma_1 * R_i), axis=2), 1 / gamma_1))  # ?x? cap-img pairs
        # posterior probabilities
        P_DQ = tf.diag_part(tf.nn.softmax(gamma_2 * R, axis=0))  # P(cap match img)
        P_QD = tf.diag_part(tf.nn.softmax(gamma_2 * R, axis=1))  # p(img match cap)
        # losses
        L1_w = -tf.reduce_mean(tf.log(P_DQ))
        L2_w = -tf.reduce_mean(tf.log(P_QD))

        ###sentence-level###
        # heatmap
        h_s = tf.nn.relu(tf.einsum('bj,cklj->bckl', e_s, v))  # pair-wise e_bar*v^T: ?x?xNx4
        # attention
        a_s = tf.einsum('bcjk,cjkl->bclk', h_s, v)  # ?x?xDx4 attnded visual reps for sen. for all pairs
        # pair-wise score
        a_s_norm = tf.nn.l2_normalize(a_s, axis=2)
        e_s_norm = tf.nn.l2_normalize(e_s, axis=1)
        R_sk = tf.einsum('bclk,bl->bck', a_s_norm, e_s_norm)  # cosine for (sen,img_reps) for all pairs
        R_s = tf.reduce_max(R_sk, axis=-1)  # ?x?
        # posterior probabilities
        P_DQ_s = tf.diag_part(tf.nn.softmax(gamma_2 * R_s, axis=0))  # P(cap match img)
        P_QD_s = tf.diag_part(tf.nn.softmax(gamma_2 * R_s, axis=1))  # P(img match cap)
        # losses
        L1_s = -tf.reduce_mean(tf.log(P_DQ_s))
        L2_s = -tf.reduce_mean(tf.log(P_QD_s))
        # overall loss
        loss = L1_w + L2_w + L1_s + L2_s

    return loss


def attn_loss(e_w, v, e_s, gamma_1, gamma_2):
    # e: ?xTxD, v: ?xNx4xD, e_bar: ?xD
    with tf.variable_scope('attention_loss'):
        ###word-level###
        # heatmap
        h = tf.nn.relu(tf.einsum('bij,cklj->bcikl', e_w, v))  # pair-wise ev^T: ?x?xTxNx4
        # attention
        a = tf.einsum('bcijl,cjlk->bcikl', h, v)  # ?x?xTxDx4 attnded visual reps for each of T words for all pairs
        # pair-wise score
        a_norm = tf.nn.l2_normalize(a, axis=3)
        e_w_norm = tf.nn.l2_normalize(e_w, axis=2)
        R_ik = tf.einsum('bcilk,bil->bcik', a_norm, e_w_norm)  # cosine for T (words,img_reps) for all pairs
        # level dropout
        # R_ik_sh = R_ik.get_shape().as_list()
        # R_ik = tf.layers.dropout(R_ik,rate=0.5,noise_shape=[1,1,1,R_ik_sh[3]],
        #                         training=isTraining)
        R_i = tf.reduce_max(R_ik, axis=-1)  # ?x?xT
        R = tf.log(tf.pow(tf.reduce_sum(tf.exp(gamma_1 * R_i), axis=2), 1 / gamma_1))  # ?x? cap-img pairs
        # posterior probabilities
        P_DQ = tf.diag_part(tf.nn.softmax(gamma_2 * R, axis=0))  # P(cap match img)
        P_QD = tf.diag_part(tf.nn.softmax(gamma_2 * R, axis=1))  # p(img match cap)
        # losses
        L1_w = -tf.reduce_mean(tf.log(P_DQ))
        L2_w = -tf.reduce_mean(tf.log(P_QD))

        ###sentence-level###
        # heatmap
        h_s = tf.nn.relu(tf.einsum('bj,cklj->bckl', e_s, v))  # pair-wise e_bar*v^T: ?x?xNx4
        # attention
        a_s = tf.einsum('bcjk,cjkl->bclk', h_s, v)  # ?x?xDx4 attnded visual reps for sen. for all pairs
        # pair-wise score
        a_s_norm = tf.nn.l2_normalize(a_s, axis=2)
        e_s_norm = tf.nn.l2_normalize(e_s, axis=1)
        R_sk = tf.einsum('bclk,bl->bck', a_s_norm, e_s_norm)  # cosine for (sen,img_reps) for all pairs
        R_s = tf.reduce_max(R_sk, axis=-1)  # ?x?
        # posterior probabilities
        P_DQ_s = tf.diag_part(tf.nn.softmax(gamma_2 * R_s, axis=0))  # P(cap match img)
        P_QD_s = tf.diag_part(tf.nn.softmax(gamma_2 * R_s, axis=1))  # P(img match cap)
        # losses
        L1_s = -tf.reduce_mean(tf.log(P_DQ_s))
        L2_s = -tf.reduce_mean(tf.log(P_QD_s))
        # overall loss
        loss = L1_w + L2_w + L1_s + L2_s

    return loss



