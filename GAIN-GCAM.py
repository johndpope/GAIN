import os
import sys
import time
from six.moves import cPickle
import numpy as np
import scipy.ndimage as nd
from PIL import Image
import tensorflow as tf
import optparse
from dataset import dataset
from crf import crf_inference

"""
GAIN-GCAM
----------------------
This code implements GAIN with the following setting:
 * Segmentation model: Grad-CAM(ICCV'17)
 * Base model: VGG16 (remove 2 max-pool)
"""

def parse_arg():
    parser = optparse.OptionParser()
    parser.add_option('-g', dest='gpu_id', default='0', help="specify to run on which GPU")
    parser.add_option('-f', dest='gpu_frac', default='0.99', help="specify the memory utilization of GPU")
    parser.add_option('-r', dest='restore_iter_id', default=None, help="continue training? default=False")
    parser.add_option('-a', dest='action', default='train', help="training or inference?")
    parser.add_option('-c', dest='with_crf', default=False, action='store_true', help="with CRF defult=False")
    (options, args) = parser.parse_args()
    return options

class GAIN():
    def __init__(self,config):
        self.config = config
        # size of image(`input`)
        self.h, self.w = self.config.get("input_size", (321,321))
        self.category_num, self.accum_num, self.with_crf = self.config.get("category_num",21), self.config.get("accum_num",1), self.config.get("with_crf",False)
        self.data, self.min_prob, self.iter_num = self.config.get("data",None), self.config.get("min_prob",1e-6), self.config.get("iter_num",0)
        self.net, self.loss, self.acc, self.saver, self.weights, self.stride = {}, {}, {}, {}, {}, {}
        self.trainable_list, self.lr_1_list, self.lr_2_list, self.lr_10_list, self.lr_20_list = [], [], [], [], []
        self.stride["input"], self.stride["input_c"] = 1, 1
        self.clip_eps = 1e-10 # clip values feeding to log function
        self.lambda_cl, self.lambda_mask_reg, self.att_th = 10, 1e-3, 0.5
        self.pre_train_epoch = 5.0 # pre-train the mask to match the cue
        self.min_lr = 1e-5
        self.agg_w, self.agg_w_bg = np.reshape(np.array([0.996**i for i in range(41*41 -1, -1, -1)]),(1,-1,1)), np.reshape(np.array([0.999**i for i in range(41*41 -1, -1, -1)]),(1,-1)) # SEC: Global-Weighted-Ranking-Pooling
        ###### Network Architecture ######
        self.opt_arch = 2 # Option 1: DeepLab | Option 2: small-DeepLab
        ###### Attention-Mining ######
        self.opt_am = 2 # Option 1: Cross Entropy | Option 2: Sum of Score (SEC[ECCV'16])
        ###### Aggregation (Pixel-Level to Image-Level) ######
        self.opt_agg = 2 # Option 1: Global Average Pooling | Option 2: Global Weighted Ranking Pooling (SEC[ECCV'16])
        ###### Classification Loss ######
        self.opt_cl = 2 # Option 1: Cross-Entropy on Aggregated Scores | Option 2: (SEC[ECCV'16])
        ###### Mask Regularizer ######
        self.opt_mask_reg = 2 # Option 1: Sum of Score | Option 2: Sum of Score Normalized by #label
    def build(self):
        if "output" not in self.net:
            with tf.name_scope("placeholder"):
                self.net["input"] = tf.placeholder(tf.float32,[None,self.h,self.w,self.config.get("input_channel",3)])
                self.net["label"] = tf.placeholder(tf.float32,[None,self.category_num])
                self.net["drop_prob"] = tf.placeholder(tf.float32)
                self.net["cues"] = tf.placeholder(tf.float32,[None,41,41,self.category_num])
            self.net["output"] = self.create_network()
        return self.net["output"]
    def create_network(self):
        if "init_model_path" in self.config: self.load_init_model()
        # Define network architecture
        if self.opt_arch==1: # Option 1: DeepLab-CRF-LargeFOV
            deeplab_layers = ["conv1_1","relu1_1","conv1_2","relu1_2","pool1",
                              "conv2_1","relu2_1","conv2_2","relu2_2","pool2",
                              "conv3_1","relu3_1","conv3_2","relu3_2","conv3_3","relu3_3","pool3",
                              "conv4_1","relu4_1","conv4_2","relu4_2","conv4_3","relu4_3","pool4",
                              "conv5_1","relu5_1","conv5_2","relu5_2","conv5_3","relu5_3","pool5",
                              "poola","fc6","relu6","drop6","fc7","relu7","drop7","fc8"]
            self.network_layers, num_cv, self.last_cv, self.dim_fmap, self.fc8_dim, self.mask_layer_name = deeplab_layers, 32, "pool5", 512, 1024, "gcam"
        elif self.opt_arch==2: # Option 2: modified DeepLab by removing several Conv+Pool layers
            small_deeplab_layers = ["conv1_1","relu1_1","conv1_2","relu1_2","pool1",
                                    "conv2_1","relu2_1","conv2_2","relu2_2","pool2",
                                    "conv3_1","relu3_1","conv3_2","relu3_2","conv3_3","relu3_3","pool3",
                                    "fc8"]
            self.network_layers, num_cv, self.last_cv, self.dim_fmap, self.fc8_dim, self.mask_layer_name = small_deeplab_layers, 17, "pool3", 256, 256, "gcam"
        # path of `input` to VGG16
        with tf.name_scope("base-cl") as scope:
            block = self.build_block("input", self.network_layers[:num_cv])
            fc = self.build_fc(block, self.network_layers[num_cv:])
            out = self.build_sp_softmax(fc, axis=3, layer_name="fc8-softmax")
            agg = self.build_aggregation(out, layer_name="fc8-agg")
            # generate the attention map with Grad-CAM
            gcam = self.build_grad_cam(target=agg, fmap=self.last_cv)
            gcam_sp = "gcam-score"
            self.net[gcam_sp] = tf.nn.sigmoid(self.net["gcam"])
            # remove discontiouous by CRF
            out = self.build_crf(gcam_sp,"input") if self.with_crf else gcam_sp
        # path of `input_c` to VGG16
        with tf.name_scope("am") as scope:
            with tf.variable_scope(tf.get_variable_scope().name, reuse=tf.AUTO_REUSE) as var_scope:
                var_scope.reuse_variables()
                # generate `input_c`, which is the complement part of the image not selected by the attention map
                input_c = self.build_input_c(gcam_sp, "input")
                block = self.build_block(input_c, self.network_layers[:num_cv], is_exist=True)
                fc = self.build_fc(block, self.network_layers[num_cv:], is_exist=True)
                out = self.build_sp_softmax(fc, is_exist=True, layer_name="fc8-softmax")
                agg = self.build_aggregation(out, layer_name="fc8-agg", is_exist=True)
        return self.net[out]
    def build_block(self, last_layer, layer_lists, is_exist=False):
        input_layer = last_layer
        for layer in layer_lists:
            player = layer if not is_exist else '-'.join([input_layer, layer])
            with tf.name_scope(layer) as scope:
                if layer.startswith("conv"):
                    self.stride[player] = self.stride[last_layer]
                    weights, bias = self.get_weights_and_bias(layer, is_exist=is_exist)
                    self.net[player] = tf.nn.conv2d(self.net[last_layer], weights, strides=[1,1,1,1], padding="SAME", name="conv") if layer[4]!="5" else tf.nn.atrous_conv2d(self.net[last_layer], weights, rate=2, padding="SAME", name="conv")
                    self.net[player] = tf.nn.bias_add(self.net[player], bias, name="bias")
                elif layer.startswith("batch_norm"):
                    self.stride[player] = self.stride[last_layer]
                    self.net[player] = tf.contrib.layers.batch_norm(self.net[last_layer])
                elif layer.startswith("relu"):
                    self.stride[player] = self.stride[last_layer]
                    self.net[player] = tf.nn.relu(self.net[last_layer],name="relu")
                elif layer.startswith("poola"):
                    self.stride[player] = self.stride[last_layer]
                    self.net[player] = tf.nn.avg_pool(self.net[last_layer], ksize=[1,3,3,1], strides=[1,1,1,1],padding="SAME",name="pool")
                elif layer.startswith("pool"):
                    c, s = (1, [1,1,1,1]) if layer[4] in ["4","5"] else (2, [1,2,2,1])
                    self.stride[player] = c*self.stride[last_layer]
                    self.net[player] = tf.nn.max_pool(self.net[last_layer],ksize=[1,3,3,1],strides=s,padding="SAME",name="pool")
                else: raise Exception("Unimplemented layer: {}".format(layer))
                last_layer = player
        return last_layer
    def build_fc(self, last_layer, layer_lists, is_exist=False):
        input_layer = last_layer.split('-')[0]
        for layer in layer_lists:
            player = layer if not is_exist else '-'.join([input_layer, layer])
            with tf.name_scope(layer) as scope:
                if layer.startswith("fc"):
                    weights, bias = self.get_weights_and_bias(layer, is_exist=is_exist)
                    if layer.startswith("fc6"): self.net[player] = tf.nn.atrous_conv2d(self.net[last_layer], weights, rate=12, padding="SAME", name="conv")
                    else: self.net[player] = tf.nn.conv2d(self.net[last_layer], weights, strides=[1,1,1,1], padding="SAME", name="conv")
                    self.net[player] = tf.nn.bias_add(self.net[player], bias, name="bias")
                elif layer.startswith("batch_norm"): self.net[player] = tf.contrib.layers.batch_norm(self.net[last_layer])
                elif layer.startswith("drop"): self.net[player] = tf.nn.dropout(self.net[last_layer], self.net["drop_prob"])
                elif layer.startswith("relu"): self.net[player] = tf.nn.relu(self.net[last_layer])
                else: raise Exception("Unimplemented layer: {}".format(layer))
                last_layer = player
        return last_layer
    def build_sp_softmax(self, last_layer, is_exist=False, axis=1, layer_name="fc8-softmax"): # SEC
        player = '-'.join([last_layer.split('-')[0], layer_name]) if is_exist else layer_name
        preds_max = tf.reduce_max(self.net[last_layer], axis=axis, keepdims=True)
        preds_exp = tf.exp(self.net[last_layer]-preds_max)
        self.net[player] = preds_exp/tf.reduce_sum(preds_exp,axis=axis, keepdims=True) + self.min_prob
        self.net[player] = self.net[player]/tf.reduce_sum(self.net[player], axis=axis, keepdims=True)
        return player
    def build_aggregation(self, last_layer, is_exist=False, layer_name="fc8-agg"):
        """Aggregate the pixel-level prediction to image-level labels"""
        player = '-'.join([last_layer.split('-')[0], layer_name]) if is_exist else layer_name
        if self.opt_agg==1: # Option 1: Global Average Pooling
            self.net[player] = tf.reduce_sum(self.net[last_layer], axis=(1,2))
        elif self.opt_agg==2: # Option 2: Global Weighted Ranking Pooling
            scores = tf.reduce_sum((self.agg_w*tf.contrib.framework.sort(tf.reshape(self.net[last_layer][:,:,:,1:],(-1,41*41,20)), axis=1))/np.sum(self.agg_w), axis=1)
            score_bg = tf.expand_dims(tf.reduce_sum((tf.contrib.framework.sort(tf.reshape(self.net[last_layer][:,:,:,0],(-1,41*41)), axis=1)*self.agg_w_bg)/np.sum(self.agg_w_bg), axis=1),axis=1)
            self.net[player] = tf.concat([score_bg, scores], axis=1)
        return player
        
    def build_crf(self, featemap_layer, img_layer, layer_name="crf"):
        def crf(featemap, image):
            crf_config = {"g_sxy":3/12,"g_compat":3,"bi_sxy":80/12,"bi_srgb":13,"bi_compat":10,"iterations":5}
            batch_size = featemap.shape[0]
            image = image.astype(np.uint8)
            ret = np.zeros(featemap.shape,dtype=np.float32)
            for i in range(batch_size): ret[i,:,:,:] = crf_inference(featemap[i], image[i], crf_config, self.category_num)
            ret = np.nan_to_num(ret) # deal with np.nan
            ret[ret<self.min_prob] = self.min_prob
            ret /= np.sum(ret,axis=3, keepdims=True)
            ret += self.clip_eps
            ret = np.log(ret)
            return ret.astype(np.float32)
        self.net[layer_name] = tf.py_func(crf, [self.net[featemap_layer], tf.image.resize_bilinear(self.net[img_layer]+self.data.img_mean, (41,41))],tf.float32) # shape [N, h, w, C]
        return layer_name
    def build_grad_cam(self, target, fmap, layer_name="gcam"):
        """
        Implement Grad-CAM(ICCV'17)
        -----------------------------------
        Input: predicted target Y[#class], feature map A[w/8,h/8]
        return: CAM[#class,w/8,h/8], where CAM[c,:,:] = ReLU(\sum_k alpha_k*A^k)
        """
        A, Y = self.net[fmap], self.net[target]*self.net["label"]
        cams = []
        for c in range(self.category_num):
            # calculate the importance of each feature map
            alpha = tf.reduce_sum(tf.gradients(Y[:,c], A)[0], axis=(1,2)) + self.clip_eps # for numerical stability
            # normalize alpha
            alpha = alpha/tf.reduce_sum(alpha, axis=1, keepdims=True)
            # linear combine the feature map to generate CAM
            cam_c = tf.reduce_sum(tf.reshape(tf.reshape(alpha, (-1,1))*tf.reshape(tf.transpose(A, [0,3,1,2]), (-1,41*41)), (-1,self.dim_fmap,41*41)), axis=1)
            cams.append(tf.nn.relu(cam_c))
        cams = tf.reshape(tf.stack(cams, axis=2), (-1,41,41,self.category_num)) + self.clip_eps # for numerical stability
        cams /= tf.reduce_sum(cams,axis=2,keepdims=True)
        self.net[layer_name] = cams
        return layer_name
    def build_input_c(self, att_layer, img_layer, layer_name="input_c"):
        """
        Generate the image complement.
        ------------------------------------------------------------------------
        Input: the image I[bsize,w,h,3], and the attention-map A[bsize,w/8,h/8,#class],
        Output: the image-complement image_c[bsize*#class,w,h,3]
        """
        image, atts = tf.image.resize_bilinear(self.net[img_layer], (self.h,self.w)), tf.image.resize_bilinear(self.net[att_layer], (self.h,self.w))
        rst, masks = [], []
        for att in tf.unstack(atts, axis=3): # generate class-specific image complement
            att = tf.cast(tf.greater(att, self.att_th), tf.float32) # threshold masking
            img_c = tf.expand_dims(image-tf.reshape(tf.reshape(image, (-1,3))*tf.reshape(att, (-1,1)), (-1,self.h,self.w,3)), axis=1)
            rst.append(img_c)
            masks.append(tf.expand_dims(1-att, axis=3))
        x = tf.concat(rst, axis=1)
        image_c = tf.reshape(x, (-1,self.h,self.w,3))
        self.net["mask"] = tf.concat(masks, axis=3)
        self.net[layer_name] = image_c
        return layer_name
    def load_init_model(self):
        """Load the pre-trained VGG16 weight"""
        model_path = self.config["init_model_path"]
        self.init_model = np.load(model_path,encoding="latin1").item()
        print("load init model success: %s" % model_path)
    def restore_from_model(self, saver, model_path, checkpoint=False):
        assert self.sess is not None
        if checkpoint: saver.restore(self.sess, tf.train.get_checkpoint_state(model_path).model_checkpoint_path)
        else: saver.restore(self.sess, model_path)
    def get_weights_and_bias(self, layer, is_exist=False):
        if is_exist: return tf.get_variable(name="{}_weights".format(layer)), tf.get_variable(name="{}_bias".format(layer))
        if layer.startswith("conv"):
            shape = [3,3,0,0]
            if layer == "conv1_1": shape[2]=3
            else:
                shape[2] = min(64*self.stride[layer], 512)
                if layer in ["conv2_1","conv3_1","conv4_1"]: shape[2]=int(shape[2]/2)
            shape[3] = min(64*self.stride[layer], 512)
        if layer.startswith("fc"):
            if layer == "fc6": shape=[3,3,512,1024]
            elif layer == "fc7": shape=[1,1,1024,1024]
            elif layer == "fc8": shape=[1,1,self.fc8_dim,self.category_num]
        if "init_model_path" not in self.config:
            weights = tf.get_variable(name="{}_weights".format(layer), initializer=tf.random_normal_initializer(stddev=0.01), shape=shape)
            bias = tf.get_variable(name="{}_bias".format(layer), initializer=tf.constant_initializer(0), shape=[shape[-1]])
        else: # restroe from init.npy
            weights = tf.get_variable(name="{}_weights".format(layer), initializer=tf.contrib.layers.xavier_initializer(uniform=True) if layer=="fc8" else tf.constant_initializer(self.init_model[layer]["w"]), shape=shape)
            bias = tf.get_variable(name="{}_bias".format(layer), initializer=tf.constant_initializer(0) if layer=="fc8" else tf.constant_initializer(self.init_model[layer]["b"]), shape = [shape[-1]])
        self.weights[layer] = (weights, bias)
        if layer != "fc8":
            self.lr_1_list.append(weights)
            self.lr_2_list.append(bias)
        else: # the lr is larger in the last layer
            self.lr_10_list.append(weights)
            self.lr_20_list.append(bias)
        self.trainable_list.append(weights)
        self.trainable_list.append(bias)
        return weights, bias
    def get_crf_loss(self):
        """Constrain the Attention Map by Conditional Random Field(NIPS'11)"""
        constrain_loss = tf.reduce_mean(tf.reduce_sum(tf.exp(self.net["crf"]) * tf.log(tf.exp(self.net["crf"])/tf.nn.sigmoid(self.net[self.mask_layer_name])+self.clip_eps), axis=3)) if self.with_crf else tf.constant(0.0)
        self.loss["constrain"] = constrain_loss
        return constrain_loss
    def get_cl_loss(self):
        """Loss of Multi-Label Classification"""
        if self.opt_cl==1: # Option 1: Cross-Entropy
            x, z = tf.reduce_sum(self.net["fc8"], axis=(1,2))[:,1:], self.net["label"][:,1:]
            return tf.reduce_mean(tf.reduce_sum(tf.maximum(x,0) - x*z + tf.log(1+tf.exp(-tf.abs(x))), axis=1))
        if self.opt_cl==2: # Option 2: SEC[ECCV'16]
            stat = tf.cast(tf.greater(self.net["label"][:,1:],0),tf.float32)
            loss_1 = -tf.reduce_mean(tf.reduce_sum(stat*tf.log(self.net["fc8-agg"][:,1:]) / tf.reduce_sum(stat,axis=1,keepdims=True), axis=1))
            loss_2 = -tf.reduce_mean(tf.reduce_sum((1-stat)*tf.log(1-tf.reduce_max(self.net["fc8-softmax"][:,:,:,1:],axis=(1,2))) / tf.reduce_sum(1-stat,axis=1,keepdims=True), axis=1))
            loss_3 = -tf.reduce_mean(tf.log(self.net["fc8-agg"][:,0]))
            return loss_1+loss_2+loss_3
    def get_pre_cue_loss(self): # SEC[ECCV'16]
        return -tf.reduce_mean(tf.reduce_sum(self.net["cues"]*tf.log(self.net["fc8-softmax"]), axis=(1,2,3), keepdims=True)/tf.reduce_sum(self.net["cues"],axis=(1,2,3), keepdims=True))
    def get_cue_loss(self):
        return -tf.reduce_mean(tf.reduce_sum(self.net["gcam-score"]*tf.log(self.net["fc8-softmax"]), axis=(1,2,3), keepdims=True)/tf.reduce_sum(self.net["gcam-score"],axis=(1,2,3), keepdims=True))
    def get_am_loss(self):
        """Implements the Attention Mining Loss described in GAIN
        ---------------------------------------------------------
        [?] Cross-Entropy
        [?] Sum of Scores by GAIN(ICCV'17)
        """
        if self.opt_am==1: # Option 1: Cross Entropy
            x = tf.reshape(tf.reduce_sum(self.net["input_c-fc8"],axis=(1,2)), (-1, self.category_num, category_num))
            x = tf.stack([x[:,c,c] for c in range(1,self.category_num)], axis=1)
            return tf.reduce_mean(tf.reduce_sum(tf.maximum(x,0)+tf.log(1+tf.exp(-tf.abs(x))), axis=1))
        if self.opt_am==2: # Option 2: Sum of Scores by SEC
            x = tf.reshape(self.net["input_c-fc8-agg"], (-1, self.category_num, category_num))
            return tf.reduce_mean(tf.reduce_sum(tf.stack([x[:,c,c] for c in range(1,self.category_num)], axis=1), axis=1) / tf.reduce_sum(self.net["label"], axis=1))
    def get_mask_reg(self):
        if self.opt_mask_reg==1: # Option 1
            return tf.reduce_mean(tf.reduce_sum(self.net["gcam-score"], axis=(1,2,3)))
        if self.opt_mask_reg==2: # Option 2
            x = tf.reduce_sum(self.net["gcam-score"][:,:,:,1:], axis=(1,2))
            return tf.reduce_mean(tf.sqrt(tf.reduce_sum(x*x, axis=1)) / tf.reduce_sum(self.net["label"], axis=1))
    def get_correct_num(self, score, label):
        return tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(tf.round(score[:,1:]), tf.round(label[:,1:])), tf.float32), axis=1))
        
    def add_loss_summary(self):
        tf.summary.scalar('cl-loss', self.loss["loss_cl"])
        tf.summary.scalar('cue-loss', self.loss["loss_cue"])
        tf.summary.scalar('am-loss', self.loss["loss_am"])
        tf.summary.scalar('crf-loss', self.loss["loss_crf"])
        tf.summary.scalar('mask-reg', self.loss["mask_reg"])
        tf.summary.scalar('input-correct', self.acc["input"])
        tf.summary.scalar('input-recall', self.acc["input-recall"])
        tf.summary.scalar('input_c-correct', self.acc["input_c"])
        tf.summary.scalar('l2', self.loss["total"]-self.loss["norm"])
        tf.summary.scalar('total', self.loss["total"])
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(os.path.join(SAVER_PATH, 'sum'))
    def optimize(self, base_lr, momentum, weight_decay):
        self.loss["loss_cue"] = self.get_cue_loss()
        self.loss["loss_cue_cl"] = self.get_pre_cue_loss()
        self.loss["loss_cl"] = self.get_cl_loss()*self.lambda_cl
        self.loss["loss_am"] = self.get_am_loss()
        self.loss["loss_crf"] = self.get_crf_loss()
        self.loss["mask_reg"] = self.get_mask_reg()*self.lambda_mask_reg
        self.loss["norm"] = self.loss["loss_cl"] + self.loss["loss_am"] + self.loss["loss_crf"] + self.loss["loss_cue"]
        self.loss["l2"] = tf.reduce_sum([tf.nn.l2_loss(self.weights[layer][0]) for layer in self.weights], axis=0)
        self.loss["total"] = self.loss["norm"] + weight_decay*self.loss["l2"] + self.loss["mask_reg"]
        self.loss["total_cl"] = self.loss["loss_cl"]+self.loss["loss_cue_cl"]+self.loss["l2"]
        self.label_num = tf.cast(tf.reduce_sum(self.net["label"][:,1:], axis=1), tf.int32)
        self.acc["input"] = tf.cast(self.get_correct_num(self.net["fc8-agg"], self.net["label"]), tf.int32)
        self.acc["input-recall"] = tf.cast(tf.reduce_mean(tf.reduce_sum(tf.round(self.net["fc8-agg"][:,1:])*self.net["label"][:,1:], axis=1)), tf.int32)
        self.acc["input_c"] = tf.cast(self.get_correct_num(self.net["input_c-fc8-agg"], tf.zeros_like(self.net["label"])), tf.int32)
        self.net["lr"] = tf.Variable(base_lr, trainable=False, dtype=tf.float32)
        # opt = tf.train.MomentumOptimizer(self.net["lr"],momentum)
        opt = tf.train.AdamOptimizer(self.net["lr"])
        # total
        gradients = opt.compute_gradients(self.loss["total"],var_list=self.trainable_list)
        self.grad, self.net["accum_gradient"], self.net["accum_gradient_accum"], new_gradients = {}, [], [], []
        for (g,v) in gradients:
            if v in self.lr_2_list: g = 2*g
            if v in self.lr_10_list: g = 10*g
            if v in self.lr_20_list: g = 20*g
            self.net["accum_gradient"].append(tf.Variable(tf.zeros_like(g),trainable=False))
            self.net["accum_gradient_accum"].append(self.net["accum_gradient"][-1].assign_add(g/self.accum_num, use_locking=True))
            new_gradients.append((self.net["accum_gradient"][-1],v))
        self.net["accum_gradient_clean"] = [g.assign(tf.zeros_like(g)) for g in self.net["accum_gradient"]]
        self.net["accum_gradient_update"]  = opt.apply_gradients(new_gradients)
        # only classification loss
        gradients = opt.compute_gradients(self.loss["total_cl"],var_list=self.trainable_list)
        self.grad, self.net["accum_gradient_cl"], self.net["accum_gradient_accum_cl"], new_gradients = {}, [], [], []
        for (g,v) in gradients:
            if v in self.lr_2_list: g = 2*g
            if v in self.lr_10_list: g = 10*g
            if v in self.lr_20_list: g = 20*g
            self.net["accum_gradient_cl"].append(tf.Variable(tf.zeros_like(g),trainable=False))
            self.net["accum_gradient_accum_cl"].append(self.net["accum_gradient_cl"][-1].assign_add(g/self.accum_num, use_locking=True))
            new_gradients.append((self.net["accum_gradient_cl"][-1],v))
        self.net["accum_gradient_clean_cl"] = [g.assign(tf.zeros_like(g)) for g in self.net["accum_gradient_cl"]]
        self.net["accum_gradient_update_cl"]  = opt.apply_gradients(new_gradients)
    def train(self, base_lr, weight_decay, momentum, batch_size, epoches, gpu_frac, patience=0.5, loss_accum_num=200):
        if not os.path.exists(PROB_PATH): os.makedirs(PROB_PATH)
        gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac))
        self.sess = tf.Session(config=gpu_options)
        x, _, y, c, id_of_image, iterator_train = self.data.next_batch(category="train",batch_size=batch_size,epoches=-1)
        self.build()
        self.optimize(base_lr,momentum, weight_decay)
        self.saver["norm"] = tf.train.Saver(max_to_keep=2,var_list=self.trainable_list)
        self.saver["lr"] = tf.train.Saver(var_list=self.trainable_list)
        self.saver["best"] = tf.train.Saver(var_list=self.trainable_list,max_to_keep=2)
        self.add_loss_summary()

        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(iterator_train.initializer)
            if self.config.get("model_path",False) is not False: self.restore_from_model(self.saver["norm"], self.config.get("model_path"), checkpoint=False)
            start_time = time.time()
            print("start_time: {}\nconfig -- lr:{} weight_decay:{} momentum:{} batch_size:{} epoches:{}".format(start_time, base_lr, weight_decay, momentum, batch_size, epoches))
            
            i, iterations_per_epoch_train = self.iter_num, self.data.get_data_len()//batch_size
            epoch = i/iterations_per_epoch_train
            accum_loss, loss_min, patience_count = [], None, 0
            while epoch < epoches:
                if i == 0: self.sess.run(tf.assign(self.net["lr"],base_lr))
                if patience_count > int(patience*iterations_per_epoch_train)-1:
                    new_lr = min(base_lr/2, self.min_lr)
                    self.saver["lr"].save(self.sess, os.path.join(self.config.get("saver_path",SAVER_PATH),"lr-%f"%base_lr), global_step=i)
                    self.sess.run(tf.assign(self.net["lr"],new_lr))
                    base_lr, patience_count = new_lr, 0
                    print('[loss saturate] reduce lr={:.5f}'.format(new_lr))
                data_x, data_y, data_c, data_id_of_image = self.sess.run([x, y, c, id_of_image])
                params = {self.net["input"]:data_x, self.net["cues"]:data_c, self.net["label"]:np.array(data_y).astype(np.float32), self.net["drop_prob"]:0.5}
                # train with only `loss_cl` for better Grad-CAM result, then train with full loss
                if epoch < self.pre_train_epoch: 
                    loss_total_tr = self.loss["total_cl"]
                    cur_loss, _ = self.sess.run([self.loss["total_cl"], self.net["accum_gradient_accum_cl"]], feed_dict=params)
                else: 
                    loss_total_tr = self.loss["total"]
                    cur_loss, _ = self.sess.run([self.loss["total"], self.net["accum_gradient_accum"]], feed_dict=params)
                loss_min = cur_loss if loss_min is None else loss_min
                accum_loss.append(cur_loss)
                if len(accum_loss)>=loss_accum_num:
                    avg_loss = np.mean(accum_loss)
                    if avg_loss > loss_min: patience_count, accum_loss = patience_count+1, []
                    else: loss_min, patience_count, accum_loss = avg_loss, 0, []
                if i % self.accum_num == self.accum_num-1:
                    # train with only `loss_cl` for better Grad-CAM result, then train with full loss
                    if epoch < self.pre_train_epoch: self.sess.run(self.net["accum_gradient_update_cl"]), self.sess.run(self.net["accum_gradient_clean_cl"])
                    else: self.sess.run(self.net["accum_gradient_update"]), self.sess.run(self.net["accum_gradient_clean"])
                if i%100 == 0:
                    summary, label_num, input_correct, input_recall, input_c_correct, loss_cl, loss_cue, loss_am, mask_reg, loss_crf, loss_l2, loss_total, lr = self.sess.run([self.merged, self.label_num, self.acc["input"], self.acc["input-recall"], self.acc["input_c"], self.loss["loss_cl"], self.loss["loss_cue"], self.loss["loss_am"], self.loss["mask_reg"], self.loss["loss_crf"], self.loss["l2"], loss_total_tr, self.net["lr"]], feed_dict=params)
                    print("{:.1f}|{}its|lr{:.5f}|label={}|{}|{}c{}|cl={:.3f}|cue={:.3f}|mask={:.3f}+{:3f}|crf={:.3f}|{:.3f}".format(epoch, i, lr, label_num[0], input_recall, input_correct, input_c_correct, loss_cl, loss_cue, loss_am, mask_reg, loss_crf, weight_decay*loss_l2, loss_total))
                    # generate mask samples
                    img_ids, pred_masks, input_c_mask, fc8_mask = self.sess.run([id_of_image, self.net[self.mask_layer_name], self.net["mask"], self.net["fc8-softmax"]], feed_dict=params)
                    self.save_masks(pred_masks, img_ids, PROB_PATH, pref=str(i))
                    self.save_masks(input_c_mask, img_ids, PROB_PATH, pref=str(i), suf='m')
                    self.save_masks(fc8_mask, img_ids, PROB_PATH, pref=str(i), suf='fc8')
                    self.save_masks(data_c, img_ids, PROB_PATH, pref=str(i), suf='cue')
                    self.writer.add_summary(summary, global_step=i)
                    self.saver["norm"].save(self.sess, os.path.join(self.config.get("saver_path",SAVER_PATH),"norm"), global_step=i)
                i+=1
                epoch = i/iterations_per_epoch_train
            end_time = time.time()
            print("end_time:{}\nduration time:{}".format(end_time, (end_time-start_time)))
    def calc_mask(self, score, eps):
        scores_exp = np.exp(score-np.max(score, axis=2, keepdims=True))
        probs = scores_exp/np.sum(scores_exp, axis=2, keepdims=True)
        probs = nd.zoom(probs, (321/probs.shape[0], 321/probs.shape[1], 1.0), order=1)
        probs = np.nan_to_num(probs)
        probs[probs<eps] = eps
        return np.argmax(probs, axis=2)
    def save_masks(self, pred_masks, img_ids, save_dir, eps=1e-5, pref=None, suf=None, calc=True):
        for idx,img_id in enumerate(img_ids):
            mask = self.calc_mask(pred_masks[idx], eps) if calc else pred_masks[idx]
            fname = img_id.decode("utf-8")
            fname = fname if suf is None else '-'.join([fname,suf])
            fname = fname if pref is None else '-'.join([pref,fname])
            cPickle.dump(mask, open('{}/{}.pkl'.format(save_dir, fname), 'wb'))
    def inference(self, gpu_frac, eps=1e-5):
        if not os.path.exists(PRED_PATH): os.makedirs(PRED_PATH)
        #Dump the predicted mask as numpy array to disk
        gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac))
        self.sess = tf.Session(config=gpu_options)
        x, gt, _, _, id_of_image, iterator_train = self.data.next_batch(batch_size=1,epoches=-1)
        self.build()
        self.saver["norm"] = tf.train.Saver(max_to_keep=2,var_list=self.trainable_list)
        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(iterator_train.initializer)
            if self.config.get("model_path",False) is not False: self.restore_from_model(self.saver["norm"], self.config.get("model_path"), checkpoint=False)
            epoch, i, iterations_per_epoch_train = 0.0, 0, self.data.get_data_len()
            while epoch < 1:
                data_x, data_gt, img_id = self.sess.run([x, gt, id_of_image])
                pred_masks = self.sess.run(self.net[self.mask_layer_name], feed_dict={self.net["input"]:data_x, self.net["drop_prob"]:0.5})
                self.save_masks(self, pred_masks, img_id, PRED_PATH, eps=eps)
                i+=1
                epoch = i/iterations_per_epoch_train


if __name__ == "__main__":
    opt = parse_arg()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    # with CRF(NIPS'11) or not
    if opt.with_crf: SAVER_PATH, PRED_PATH, PROB_PATH = "gain_gcam_crf-saver", "gain_gcam_crf-preds", "gain_gcam_crf-probs"
    else: SAVER_PATH, PRED_PATH, PROB_PATH = "gain_gcam-saver", "gain_gcam-preds", "gain_gcam-probs"
    # actual batch size=batch_size*accum_num
    batch_size, input_size, category_num, epoches = 1, (321,321), 21, 20
    category = "train+val" if opt.action == 'inference' else "train"
    data = dataset({"batch_size":batch_size, "input_size":input_size, "epoches":epoches, "category_num":category_num, "categorys":[category]})
    if opt.restore_iter_id == None: gain = GAIN({"data":data, "batch_size":batch_size, "input_size":input_size, "epoches":epoches, "category_num":category_num, "init_model_path":"./model/init.npy", "accum_num":16, "with_crf":opt.with_crf})
    else: gain = GAIN({"data":data, "batch_size":batch_size, "input_size":input_size, "epoches":epoches, "category_num":category_num, "model_path":"{}/norm-{}".format(SAVER_PATH, opt.restore_iter_id), "accum_num":16, "with_crf":opt.with_crf, "iter_num":int(opt.restore_iter_id)})
    if opt.action == 'train': gain.train(base_lr=1e-3, weight_decay=5e-5, momentum=0.9, batch_size=batch_size, epoches=epoches, gpu_frac=float(opt.gpu_frac))
    elif opt.action == 'inference': gain.inference(gpu_frac=float(opt.gpu_frac))