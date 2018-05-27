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
from GAIN import GAIN

"""
MinMax-GAIN
------------
 * Classification Model: DeepLab + SEC-GWRPool Aggregation
 * Refining Model: DeConv
"""

def parse_arg():
    """Utility function for the convenience of parsing arguments from commands"""
    parser = optparse.OptionParser()
    parser.add_option('-g', dest='gpu_id', default='0', help='specify to run on which GPU')
    parser.add_option('-f', dest='gpu_frac', default='0.49', help='specify the memory utilization of GPU')
    parser.add_option('-r', dest='restore_iter_id', default=None, help="continue training? default=False")
    parser.add_option('-a', dest='action', default='train', help="training or inference?")
    parser.add_option('-p', dest='pre', default=None, help="pre-fix to append to the saving directory")
    (options, args) = parser.parse_args()
    return options

class MinMaxGAIN(GAIN):
    def __init__(self,config):
        GAIN.__init__(self,config)
        self.stride["reshaped_att"] = 1
        # hyperparameter for weighting the regularization terms
        self.lambda_l2, self.lambda_mask_reg = 5e-5, 1e-3
    def create_network(self):
        if "init_model_path" in self.config: self.load_init_model()
        # Define network architecture
        deeplab_layers = ["conv1_1","relu1_1","conv1_2","relu1_2","pool1",
                          "conv2_1","relu2_1","conv2_2","relu2_2","pool2",
                          "conv3_1","relu3_1","conv3_2","relu3_2","conv3_3","relu3_3","pool3",
                          "conv4_1","relu4_1","conv4_2","relu4_2","conv4_3","relu4_3","pool4",
                          "conv5_1","relu5_1","conv5_2","relu5_2","conv5_3","relu5_3","pool5",
                          "poola","fc6","relu6","drop6","fc7","relu7","drop7","fc8"]
        self.network_layers, num_cv, self.last_cv, self.dv_dim = deeplab_layers, 32, "pool5", 256
        # path of `input` to image-level score
        with tf.name_scope("cl-model") as scope:
            block = self.build_block("input", self.network_layers[:num_cv])
            fc = self.build_fc(block, self.network_layers[num_cv:])
            softmax = self.build_sp_softmax(fc) # SEC: `fc8-softmax` is our attention map
            agg = self.build_aggregation(softmax)
            crf = self.build_crf(fc,"input",fmap_size=(41,41),layer_name="att-crf") # remove discontinuity by CRF(NIPS'11)
        # resize/refine attention(`fc8-softmax`) to the final prediction mask(`input` size)
        with tf.name_scope("mask-model") as scope:
            mask = self.build_mask(softmax)
            crf = self.build_crf(mask,"input",fmap_size=(self.h,self.w),layer_name="mask-crf") # remove discontinuity by CRF(NIPS'11)
        # Take the masked `input_m` and unmasked `input_c` part of the given image
        input_m, input_c = self.separate_img_by_mask("input", mask)
        # path of `input_c` to image-level score
        with tf.name_scope("am") as scope:
            with tf.variable_scope(tf.get_variable_scope().name, reuse=tf.AUTO_REUSE) as var_scope:
                var_scope.reuse_variables()
                block = self.build_block(input_c, self.network_layers[:num_cv], is_exist=True)
                fc = self.build_fc(block, self.network_layers[num_cv:], is_exist=True)
                softmax = self.build_sp_softmax(fc, is_exist=True)
                agg = self.build_aggregation(softmax, is_exist=True) # image-level score given `input_c`
        return self.net[crf]
    def build_block(self, last_layer, layer_lists, is_exist=False, tr_list="cl"):
        """Directly taken from (xtudbxk's repository)[xtudbxk/SEC-tensorflow]"""
        input_layer = last_layer
        for layer in layer_lists:
            player = layer if not is_exist else '-'.join([input_layer, layer])
            with tf.name_scope(layer) as scope:
                if layer.startswith("conv"):
                    self.stride[player] = self.stride[last_layer]
                    weights, bias = self.get_weights_and_bias(layer, is_exist=is_exist, tr_list=tr_list)
                    self.net[player] = tf.nn.conv2d(self.net[last_layer], weights, strides=[1,1,1,1], padding="SAME", name="conv") if layer[4]!="5" else tf.nn.atrous_conv2d(self.net[last_layer], weights, rate=2, padding="SAME", name="conv")
                    self.net[player] = tf.nn.bias_add(self.net[player], bias, name="bias")
                elif layer.startswith("dconv"): # For Refining-Model, implemented by De-Convolution Layers
                    out_shape = {"dconv1":[self.bsize*self.category_num,41*2,41*2,self.dv_dim], "dconv2":[self.bsize*self.category_num,41*4,41*4,self.dv_dim], "dconv3":[self.bsize*self.category_num,41*8,41*8,1]}
                    self.stride[player] = self.stride[last_layer]
                    weights, bias = self.get_weights_and_bias(player, is_exist=is_exist, tr_list=tr_list)
                    self.net[player] = tf.nn.conv2d_transpose(self.net[last_layer], weights, out_shape[layer], strides=[1,2,2,1], padding='SAME', name="dconv")
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
    def build_mask(self, att_layer, layer_name="mask"):
        """
        The De-Convolution Network
        ---------------------------
        Input: Masks of size 41x41 for all the 21 classes
        Output: Refined Masks of size 321x321 for all the 21 classes
        """
        self.net["reshaped_att"] = tf.reshape(tf.transpose(self.net[att_layer], [0,3,1,2]), (-1,41,41,1))
        self.build_block("reshaped_att", ["dconv1","batch_norm_dcv1","relu_dcv1","dconv2","batch_norm_dcv2","relu_dcv2","dconv3","batch_norm_dcv3","relu_dcv3"], tr_list="re")
        mask_output = tf.transpose(tf.reshape(self.net["relu_dcv3"], (self.bsize,self.category_num,41*8,41*8)), [0,2,3,1])
        mask_output = tf.image.resize_bilinear(mask_output, (self.h,self.w))
        self.net[layer_name] = tf.nn.sigmoid(self.mask_coeff*(mask_output-self.mask_th))
        return layer_name
    def separate_img_by_mask(self, image_layer, mask_layer, layer_names=["input_m","input_c"]):
        """
        Separate the image by the mask.
        ------------------------------------------------------------------------
        Input: image `I[bsize,w,h,3]` and mask `A[bsize,w,h,category_num]`
        Output: masked image `image_m[bsize*category_num,w,h,3]` and unmasked image `image_c[bsize*category_num,w,h,3]`
        """
        image, masks = self.net[image_layer], self.net[mask_layer]
        img_ms, img_cs = [], []
        for mask in tf.unstack(masks, axis=3): # separate the image by the mask of each class A[:,:,:,c]
            img_m = tf.expand_dims(tf.reshape(tf.multiply(tf.reshape(image, (-1,3)), tf.reshape(mask, (-1,1))), (-1,self.w,self.h,3)), axis=1)
            img_c = tf.expand_dims(image-tf.reshape(tf.multiply(tf.reshape(image, (-1,3)), tf.reshape(mask, (-1,1))), (-1,self.w,self.h,3)), axis=1)
            img_ms.append(img_m)
            img_cs.append(img_c)
        self.net[layer_names[0]] = tf.reshape(tf.stack(img_ms, axis=1), (-1,self.w,self.h,3))
        self.net[layer_names[1]] = tf.reshape(tf.stack(img_cs, axis=1), (-1,self.w,self.h,3))
        return layer_names
    def get_weights_and_bias(self, layer, is_exist=False, tr_list="cl"):
        if is_exist: return tf.get_variable(name="{}_weights".format(layer)), tf.get_variable(name="{}_bias".format(layer))
        if layer.startswith("conv"):
            shape = [3,3,0,0]
            if layer == "conv1_1": shape[2]=3
            else:
                shape[2] = min(64*self.stride[layer], 512)
                if layer in ["conv2_1","conv3_1","conv4_1"]: shape[2]=int(shape[2]/2)
            shape[3] = min(64*self.stride[layer], 512)
        if layer.startswith("dconv"): # For Refining-Model, implemented by De-Convolution Layers
            shape = [3,3,0,0]
            if layer == "dconv1": shape[2], shape[3] = self.dv_dim, 1
            elif layer == "dconv2": shape[2], shape[3] = self.dv_dim, self.dv_dim
            elif layer == "dconv3": shape[2], shape[3] = 1, self.dv_dim
        if layer.startswith("fc"):
            if layer == "fc6": shape=[3,3,512,1024]
            elif layer == "fc7": shape=[1,1,1024,1024]
            elif layer == "fc8": shape=[1,1,1024,self.category_num]
        if "init_model_path" not in self.config:
            weights = tf.get_variable(name="{}_weights".format(layer), initializer=tf.random_normal_initializer(stddev=0.01), shape=shape)
            bias = tf.get_variable(name="{}_bias".format(layer), initializer=tf.constant_initializer(0), shape = [shape[-1] if not layer.startswith("dconv") else shape[-2]])
        else: # restroe from init.npy
            not_VGG16_layers = (layer == "fc8" or layer.startswith("dconv"))
            weights = tf.get_variable(name="{}_weights".format(layer), initializer=tf.contrib.layers.xavier_initializer(uniform=True) if not_VGG16_layers else tf.constant_initializer(self.init_model[layer]["w"]), shape=shape)
            bias = tf.get_variable(name="{}_bias".format(layer), initializer=tf.constant_initializer(0) if not_VGG16_layers else tf.constant_initializer(self.init_model[layer]["b"]), shape = [shape[-1] if not layer.startswith("dconv") else shape[-2]])
        self.weights[layer] = (weights, bias)
        if layer != "fc8":
            self.lr_1_list.append(weights)
            self.lr_2_list.append(bias)
        else: # the lr is larger in the last layer
            self.lr_10_list.append(weights)
            self.lr_20_list.append(bias)
        if tr_list=="cl":
            self.cl_list.append(weights)
            self.cl_list.append(bias)
        else:
            self.re_list.append(weights)
            self.re_list.append(bias)
        return weights, bias
    def get_cue_loss(self):
        """SEC training loss (directly taken from xtudbxk's implementation for SEC[ECCV'16])"""
        return -tf.reduce_mean(tf.reduce_sum(self.net["cues"]*tf.log(self.net["fc8-softmax"]), axis=(1,2,3), keepdims=True)/tf.reduce_sum(self.net["cues"],axis=(1,2,3), keepdims=True))
    def get_cl_loss(self):
        """SEC training loss (directly taken from xtudbxk's implementation for SEC[ECCV'16])"""
        stat_2d = tf.cast(tf.greater(self.net["label"][:,1:], 0), tf.float32)
        self.loss_1 = -tf.reduce_mean(tf.reduce_sum((stat_2d*tf.log(self.net["fc8-agg"][:,1:]) / tf.reduce_sum(stat_2d,axis=1,keepdims=True)), axis=1))
        self.loss_2 = -tf.reduce_mean(tf.reduce_sum((1-stat_2d)*tf.log(1-tf.reduce_max(self.net["fc8-softmax"][:,:,:,1:],axis=(1,2))) / tf.reduce_sum(1-stat_2d,axis=1,keepdims=True), axis=1))
        self.loss_3 = -tf.reduce_mean(tf.log(self.net["fc8-agg"][:,0]))  
        return self.loss_1+self.loss_2+self.loss_3
    def get_re_loss(self):
        stat_2d = tf.cast(tf.greater(self.net["label"][:,1:], 0), tf.float32)
        self.loss_1 = -tf.reduce_mean(tf.reduce_sum((stat_2d*tf.log(self.net["input_c-fc8-agg"][:,1:]) / tf.reduce_sum(stat_2d,axis=1,keepdims=True)), axis=1))
        self.loss_2 = -tf.reduce_mean(tf.reduce_sum((1-stat_2d)*tf.log(1-tf.reduce_max(self.net["input_c-fc8-softmax"][:,:,:,1:],axis=(1,2))) / tf.reduce_sum(1-stat_2d,axis=1,keepdims=True), axis=1))
        self.loss_3 = -tf.reduce_mean(tf.log(self.net["input_c-fc8-agg"][:,0]))  
        return self.loss_1+self.loss_2+self.loss_3
    def get_crf_loss(self, score_layer, crf_layer):
        return tf.reduce_mean(tf.reduce_sum(tf.exp(self.net[crf_layer]) * tf.log(tf.exp(self.net[crf_layer])/self.net[score_layer]), axis=3))
    def get_mask_reg(self):
        x = tf.reduce_sum(self.net["mask"][:,:,:,1:], axis=(1,2))
        return tf.reduce_mean(tf.sqrt(tf.reduce_sum(x*x, axis=1)) / tf.reduce_sum(self.net["label"], axis=1))
    def add_loss_summary(self):
        for el in ["obj_cl","obj_re","loss_cue","loss_cl","loss_re","loss_att_crf","loss_mask_crf","l2","mask_reg"]:
            tf.summary.scalar(el, self.loss[el])
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(os.path.join(self.SAVER_PATH, 'sum'))
    def optimize(self, base_lr, momentum):
        self.loss["loss_cue"] = self.get_cue_loss()
        self.loss["loss_cl"] = self.get_cl_loss()
        self.loss["loss_re"] = self.get_re_loss()
        # regularizer
        self.loss["loss_att_crf"] = self.get_crf_loss("fc8-softmax","att-crf") # CRF[NIPS'11]
        self.loss["loss_mask_crf"] = self.get_crf_loss("mask","mask-crf") # CRF[NIPS'11]
        self.loss["l2"] = tf.reduce_sum([tf.nn.l2_loss(self.weights[layer][0]) for layer in self.weights], axis=0)*self.lambda_l2
        self.loss["mask_reg"] = self.get_mask_reg()*self.lambda_mask_reg
        # objectives
        self.loss["obj_cl"] = self.loss["loss_cue"] + self.loss["loss_cl"] + self.loss["loss_att_crf"] + self.loss["l2"]
        self.loss["obj_re"] = -self.loss["loss_re"] + self.loss["loss_mask_crf"]+ self.loss["l2"] + self.loss["mask_reg"]
        self.net["lr"] = tf.Variable(base_lr, trainable=False, dtype=tf.float32)
        opt = tf.train.MomentumOptimizer(self.net["lr"],momentum)
        cl_gradients = opt.compute_gradients(self.loss["obj_cl"],var_list=self.cl_list)
        re_gradients = opt.compute_gradients(self.loss["obj_re"],var_list=self.re_list)
        for suf,grads in zip(["cl","re"],[cl_gradients,re_gradients]):
            ag, agacc, agclean, agupdate = '_'.join(["accum_gradient",suf]), '_'.join(["accum_gradient_accum",suf]), '_'.join(["accum_gradient_clean",suf]), '_'.join(["accum_gradient_update",suf])
            self.grad, self.net[ag], self.net[agacc], new_gradients = {}, [], [], []
            for (g,v) in grads:
                if v in self.lr_2_list: g = 2*g
                if v in self.lr_10_list: g = 10*g
                if v in self.lr_20_list: g = 20*g
                self.net[ag].append(tf.Variable(tf.zeros_like(g),trainable=False))
                self.net[agacc].append(self.net[ag][-1].assign_add(g/self.accum_num, use_locking=True))
                new_gradients.append((self.net[ag][-1],v))
            self.net[agclean] = [g.assign(tf.zeros_like(g)) for g in self.net[ag]]
            self.net[agupdate]  = opt.apply_gradients(new_gradients)
    def train(self, base_lr, momentum, batch_size, epoches, gpu_frac):
        if not os.path.exists(self.PROB_PATH): os.makedirs(self.PROB_PATH)
        gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac))
        self.sess = tf.Session(config=gpu_options)
        x, _, y, c, id_of_image, iterator_train = self.data.next_batch(category="train",batch_size=batch_size,epoches=-1)
        self.build()
        self.optimize(base_lr,momentum)
        self.trainable_list = self.cl_list+self.re_list
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
            print("start_time: {}\nconfig -- lr:{} momentum:{} batch_size:{} epoches:{}".format(start_time, base_lr, momentum, batch_size, epoches))
            
            epoch, i, iterations_per_epoch_train = 0, 0, self.data.get_data_len()//batch_size
            while epoch < epoches:
                if i == 0: self.sess.run(tf.assign(self.net["lr"],base_lr))
                if i == 10*iterations_per_epoch_train:
                    new_lr = 1e-4
                    self.saver["lr"].save(self.sess, os.path.join(self.SAVER_PATH,"lr-%f"%base_lr), global_step=i)
                    self.sess.run(tf.assign(self.net["lr"], new_lr))
                    base_lr = new_lr
                if i == 20*iterations_per_epoch_train:
                    new_lr = 1e-5
                    self.saver["lr"].save(self.sess, os.path.join(self.SAVER_PATH,"lr-%f"%base_lr), global_step=i)
                    self.sess.run(tf.assign(self.net["lr"],new_lr))
                    base_lr = new_lr
                # get current batch
                data_x, data_y, data_c, data_id_of_image = self.sess.run([x, y, c, id_of_image])
                params = {self.net["input"]:data_x, self.net["cues"]:data_c, self.net["label"]:np.array(data_y).astype(np.float32), self.net["drop_prob"]:0.5}
                # update Classification Model
                self.sess.run(self.net["accum_gradient_accum_cl"], feed_dict=params)
                # update Refining Model
                self.sess.run(self.net["accum_gradient_accum_re"], feed_dict=params)
                if i % self.accum_num == self.accum_num-1: 
                    _, _ = self.sess.run(self.net["accum_gradient_update_cl"]), self.sess.run(self.net["accum_gradient_clean_cl"])
                    _, _ = self.sess.run(self.net["accum_gradient_update_re"]), self.sess.run(self.net["accum_gradient_clean_re"])
                if i%100 == 0:
                    summary, obj_cl, obj_re, lr = self.sess.run([self.merged, self.loss["obj_cl"], self.loss["obj_re"], self.net["lr"]], feed_dict=params)
                    print("{:.1f}th|{}its|lr={:.5f}|obj_cl={:.5f}|obj_re={:.5f}".format(epoch, i, lr, obj_cl, obj_re))
                    # generate mask samples
                    img_ids, att_mask, re_mask = self.sess.run([id_of_image, self.net["fc8-softmax"], self.net["mask"]], feed_dict=params)
                    self.save_masks(att_mask, img_ids, self.PROB_PATH, pref=str(i), suf='fc8')
                    self.save_masks(re_mask, img_ids, self.PROB_PATH, pref=str(i), suf='re')
                    self.save_masks(data_c, img_ids, self.PROB_PATH, pref=str(i), suf='cue')
                    self.writer.add_summary(summary, global_step=i)
                if i%3000 == 2999:
                    self.saver["norm"].save(self.sess, os.path.join(self.SAVER_PATH,"norm"), global_step=i)
                i+=1
                epoch = i/iterations_per_epoch_train
            end_time = time.time()
            print("end_time:{}\nduration time:{}".format(end_time, (end_time-start_time)))
    def inference(self, gpu_frac, eps=1e-5):
        if not os.path.exists(self.PRED_PATH): os.makedirs(self.PRED_PATH)
        # Dump the predicted mask as numpy array to disk
        gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac))
        self.sess = tf.Session(config=gpu_options)
        x, gt, _, c, id_of_image, iterator_train = self.data.next_batch(batch_size=1,epoches=-1)
        self.build()
        self.saver["norm"] = tf.train.Saver(max_to_keep=2,var_list=self.cl_list+self.re_list)
        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(iterator_train.initializer)
            if self.config.get("model_path",False) is not False: self.restore_from_model(self.saver["norm"], self.config.get("model_path"), checkpoint=False)
            epoch, i, iterations_per_epoch_train = 0.0, 0, self.data.get_data_len()
            while epoch < 1:
                data_x, data_c, data_gt, img_ids = self.sess.run([x, c, gt, id_of_image])
                att_mask, re_mask = self.sess.run([self.net["fc8-softmax"], self.net["mask"]], feed_dict={self.net["input"]:data_x, self.net["drop_prob"]:0.5})
                self.save_masks(att_mask, img_ids, self.PRED_PATH, pref='10epoch', suf='fc8')
                self.save_masks(re_mask, img_ids, self.PRED_PATH, pref='10epoch', suf='re')
                self.save_masks(data_c, img_ids, self.PRED_PATH, pref='10epoch', suf='cue')
                i+=1
                epoch = i/iterations_per_epoch_train


if __name__ == "__main__":
    opt = parse_arg()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    batch_size, input_size, category_num, epoches = 1, (321,321), 21, 10
    SAVER_PATH, PRED_PATH, PROB_PATH = "saver", "preds", "probs"
    if opt.pre is not None:
        SAVER_PATH, PRED_PATH, PROB_PATH = '-'.join([opt.pre,SAVER_PATH]), '-'.join([opt.pre,PRED_PATH]), '-'.join([opt.pre,PROB_PATH])
    
    data = dataset({"batch_size":batch_size, "input_size":input_size, "epoches":epoches, "category_num":category_num, "categorys":["train"]})
    if opt.restore_iter_id == None: gain = GAIN({"data":data, "batch_size":batch_size, "input_size":input_size, "epoches":epoches, "category_num":category_num, "init_model_path":"./model/init.npy", "accum_num":16, "paths":[SAVER_PATH, PRED_PATH, PROB_PATH]})
    else: gain = GAIN({"data":data, "batch_size":batch_size, "input_size":input_size, "epoches":epoches, "category_num":category_num, "model_path":"{}/norm-{}".format(SAVER_PATH, opt.restore_iter_id), "accum_num":16, "paths":[SAVER_PATH, PRED_PATH, PROB_PATH]})
    if opt.action == 'train':
        gain.train(base_lr=1e-3, momentum=0.9, batch_size=batch_size, epoches=epoches, gpu_frac=float(opt.gpu_frac))
    elif opt.action == 'inference':
        gain.inference(gpu_frac=float(opt.gpu_frac))