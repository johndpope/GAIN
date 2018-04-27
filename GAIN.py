import os
import sys
import time
import numpy as np
import tensorflow as tf
import optparse
from dataset import dataset
from crf import crf_inference

SAVER_PATH = "gain-saver"

def parse_arg():
    parser = optparse.OptionParser()
    parser.add_option('-g', dest='gpu_id', default='0', help='specify to run on which GPU')
    parser.add_option('-f', dest='gpu_frac', default='0.49', help='specify the memory utilization of GPU')
    parser.add_option('-r', dest='restore_iter_id', default=None, help="continue training? default=False")
    (options, args) = parser.parse_args()
    return options

class GAIN():
    def __init__(self,config):
        self.config = config
        self.h, self.w = self.config.get("input_size", (321,321))
        self.cw, self.ch = 321,321
        self.category_num, self.accum_num = self.config.get("category_num",21), self.config.get("accum_num",1)
        self.data, self.min_prob = self.config.get("data",None), self.config.get("min_prob",0.0001)
        self.net, self.loss, self.saver, self.weights, self.stride = {}, {}, {}, {}, {}
        self.trainable_list, self.lr_1_list, self.lr_2_list, self.lr_10_list, self.lr_20_list = [], [], [], [], []
        self.stride["input"] = 1
        self.stride["input_c"] = 1

    def build(self):
        if "output" not in self.net:
            with tf.name_scope("placeholder"):
                self.net["input"] = tf.placeholder(tf.float32,[None,self.h,self.w,self.config.get("input_channel",3)])
                self.net["label"] = tf.placeholder(tf.int32,[None,self.category_num])
                self.net["cues"] = tf.placeholder(tf.float32,[None,41,41,self.category_num])
                self.net["drop_prob"] = tf.placeholder(tf.float32)
            self.net["output"] = self.create_network()
        return self.net["output"]
    def create_network(self):
        if "init_model_path" in self.config: self.load_init_model()
        with tf.name_scope("deeplab") as scope:
            block = self.build_block("input", ["conv1_1","relu1_1","conv1_2","relu1_2","pool1",
                                              "conv2_1","relu2_1","conv2_2","relu2_2","pool2",
                                              "conv3_1","relu3_1","conv3_2","relu3_2","conv3_3","relu3_3","pool3",
                                              "conv4_1","relu4_1","conv4_2","relu4_2","conv4_3","relu4_3","pool4",
                                              "conv5_1","relu5_1","conv5_2","relu5_2","conv5_3","relu5_3","pool5","pool5a"])
            fc = self.build_fc(block, ["fc6","relu6","drop6","fc7","relu7","drop7","fc8"])
        with tf.name_scope("sec") as scope:
            softmax, crf = self.build_sp_softmax(fc), self.build_crf(fc,"input")
        with tf.name_scope("am") as scope:
            with tf.variable_scope(tf.get_variable_scope().name, reuse=tf.AUTO_REUSE) as var_scope:
                var_scope.reuse_variables()
                input_c = self.build_input_c("fc8-softmax", "input")
                block = self.build_block(input_c, ["conv1_1","relu1_1","conv1_2","relu1_2","pool1",
                                                   "conv2_1","relu2_1","conv2_2","relu2_2","pool2",
                                                   "conv3_1","relu3_1","conv3_2","relu3_2","conv3_3","relu3_3","pool3",
                                                   "conv4_1","relu4_1","conv4_2","relu4_2","conv4_3","relu4_3","pool4",
                                                   "conv5_1","relu5_1","conv5_2","relu5_2","conv5_3","relu5_3","pool5","pool5a"], is_exist=True)
                fc = self.build_fc(block, ["fc6","relu6","drop6","fc7","relu7","drop7","fc8"], is_exist=True)
                softmax = self.build_sp_softmax(fc, is_exist=True)
        return self.net[crf]
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
                elif layer.startswith("pool5a"):
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
                    self.net[player] = tf.nn.atrous_conv2d(self.net[last_layer], weights, rate=12, padding="SAME", name="conv") if layer.startswith("fc6") else tf.nn.conv2d(self.net[last_layer], weights, strides=[1,1,1,1], padding="SAME", name="conv")
                    self.net[player] = tf.nn.bias_add(self.net[player], bias, name="bias")
                elif layer.startswith("batch_norm"): self.net[player] = tf.contrib.layers.batch_norm(self.net[last_layer])
                elif layer.startswith("drop"): self.net[player] = tf.nn.dropout(self.net[last_layer], self.net["drop_prob"])
                elif layer.startswith("relu"): self.net[player] = tf.nn.relu(self.net[last_layer])
                else: raise Exception("Unimplemented layer: {}".format(layer))
                last_layer = player
        return last_layer
    def build_sp_softmax(self, last_layer, is_exist=False):
        layer = "fc8-softmax"
        player = '-'.join([last_layer.split('-')[0], layer]) if is_exist else layer
        preds_max = tf.reduce_max(self.net[last_layer], axis=3, keepdims=True)
        preds_exp = tf.exp(self.net[last_layer]-preds_max)
        self.net[player] = preds_exp/tf.reduce_sum(preds_exp,axis=3, keepdims=True) + self.min_prob
        self.net[player] = self.net[player]/tf.reduce_sum(self.net[player], axis=3, keepdims=True)
        return player
    def build_crf(self, featemap_layer, img_layer):
        def crf(featemap, image):
            crf_config = {"g_sxy":3/12,"g_compat":3,"bi_sxy":80/12,"bi_srgb":13,"bi_compat":10,"iterations":5}
            batch_size = featemap.shape[0]
            image = image.astype(np.uint8)
            ret = np.zeros(featemap.shape,dtype=np.float32)
            for i in range(batch_size):
                ret[i,:,:,:] = crf_inference(featemap[i], image[i], crf_config, self.category_num)
            ret[ret<self.min_prob] = self.min_prob
            ret /= np.sum(ret,axis=3, keepdims=True)
            ret = np.log(ret)
            return ret.astype(np.float32)
        self.net["crf"] = tf.py_func(crf, [self.net[featemap_layer], tf.image.resize_bilinear(self.net[img_layer]+self.data.img_mean, (41,41))],tf.float32) # shape [N, h, w, C]
        return "crf"
    def build_input_c(self, att_layer, img_layer):
        image, atts = tf.image.resize_bilinear(self.net[img_layer], (self.cw,self.ch)), tf.image.resize_bilinear(self.net[att_layer], (self.cw,self.ch))
        layer = "input_c"
        rst = []
        for att in tf.unstack(atts, axis=3):
            rst.append(tf.expand_dims(image-tf.reshape(tf.multiply(tf.reshape(image, (-1,3)), tf.reshape(att, (-1,1))), (-1,self.cw,self.ch,3)), axis=1))
        x = tf.stack(rst, axis=1)
        image_c = tf.reshape(x, (-1,self.cw,self.ch,3))
        self.net[layer] = image_c
        return layer
    def load_init_model(self):
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
            elif layer == "fc8": shape=[1,1,1024,self.category_num]
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

    def get_cl_loss(self):
        # seed
        seed_loss = -tf.reduce_mean(tf.reduce_sum(self.net["cues"]*tf.log(self.net["fc8-softmax"]), axis=(1,2,3), keepdims=True)/tf.reduce_sum(self.net["cues"],axis=(1,2,3), keepdims=True))
        # expand
        stat, probs_bg, probs = self.net["label"][:,1:], self.net["fc8-softmax"][:,:,:,0], self.net["fc8-softmax"][:,:,:,1:]
        weights, weights_bg, stat_2d = np.reshape(np.array([0.996**i for i in range(41*41 -1, -1, -1)]),(1,-1,1)), np.reshape(np.array([0.999**i for i in range(41*41 -1, -1, -1)]),(1,-1)), tf.cast(tf.greater(stat, 0), tf.float32)
        self.loss_1 = -tf.reduce_mean(tf.reduce_sum((stat_2d*tf.log(tf.reduce_sum((tf.contrib.framework.sort(tf.reshape(probs,(-1,41*41,20)), axis=1)*weights)/np.sum(weights), axis=1)) / tf.reduce_sum(stat_2d,axis=1,keepdims=True)), axis=1))
        self.loss_2 = -tf.reduce_mean(tf.reduce_sum(((1-stat_2d)*tf.log(1-tf.reduce_max(probs,axis=(1,2))) / tf.reduce_sum((1-stat_2d),axis=1,keepdims=True)), axis=1))
        self.loss_3 = -tf.reduce_mean(tf.log(tf.reduce_sum((tf.contrib.framework.sort(tf.reshape(probs_bg,(-1,41*41)), axis=1)*weights_bg)/np.sum(weights_bg), axis=1)))
        expand_loss = self.loss_1+self.loss_2+self.loss_3
        # constrain
        constrain_loss = tf.reduce_mean(tf.reduce_sum(tf.exp(self.net["crf"]) * tf.log(tf.exp(self.net["crf"])/self.net["fc8-softmax"]), axis=3))
        self.loss["seed"] = seed_loss
        self.loss["expand"] = expand_loss
        self.loss["constrain"] = constrain_loss
        loss = seed_loss+expand_loss+constrain_loss
        return loss
    
    def get_am_loss(self):
        w, h = int((self.cw+7)/8), int((self.ch+7)/8)
        return tf.reduce_mean(tf.reduce_sum(tf.reshape(self.net["input_c-fc8-softmax"], (-1,(category_num)*(category_num)*w*h)), axis=1)/tf.cast(tf.reduce_sum(self.net["label"], axis=1), tf.float32))

    def optimize(self, base_lr, momentum, weight_decay):
        self.loss["loss_cl"] = self.get_cl_loss()
        self.loss["loss_am"] = self.get_am_loss()
        self.loss["norm"] = self.loss["loss_cl"] + self.loss["loss_am"]
        self.loss["l2"] = tf.reduce_sum([tf.nn.l2_loss(self.weights[layer][0]) for layer in self.weights], axis=0)
        self.loss["total"] = self.loss["norm"] + weight_decay*self.loss["l2"]
        self.net["lr"] = tf.Variable(base_lr, trainable=False, dtype=tf.float32)
        opt = tf.train.MomentumOptimizer(self.net["lr"],momentum)
        gradients = opt.compute_gradients(self.loss["total"],var_list=self.trainable_list)
        self.grad = {}
        self.net["accum_gradient"] = []
        self.net["accum_gradient_accum"] = []
        new_gradients = []
        for (g,v) in gradients:
            if v in self.lr_2_list: g = 2*g
            if v in self.lr_10_list: g = 10*g
            if v in self.lr_20_list: g = 20*g
            self.net["accum_gradient"].append(tf.Variable(tf.zeros_like(g),trainable=False))
            self.net["accum_gradient_accum"].append(self.net["accum_gradient"][-1].assign_add(g/self.accum_num, use_locking=True))
            new_gradients.append((self.net["accum_gradient"][-1],v))

        self.net["accum_gradient_clean"] = [g.assign(tf.zeros_like(g)) for g in self.net["accum_gradient"]]
        self.net["accum_gradient_update"]  = opt.apply_gradients(new_gradients)

    def train(self, base_lr, weight_decay, momentum, batch_size, epoches, gpu_frac):
        gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_frac))
        self.sess = tf.Session(config=gpu_options)
        x, _, y, c, id_of_image, iterator_train = self.data.next_batch(category="train",batch_size=batch_size,epoches=-1)
        self.build()
        self.optimize(base_lr,momentum, weight_decay)
        self.saver["norm"] = tf.train.Saver(max_to_keep=2,var_list=self.trainable_list)
        self.saver["lr"] = tf.train.Saver(var_list=self.trainable_list)
        self.saver["best"] = tf.train.Saver(var_list=self.trainable_list,max_to_keep=2)

        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(iterator_train.initializer)
            if self.config.get("model_path",False) is not False: self.restore_from_model(self.saver["norm"], self.config.get("model_path"), checkpoint=False)
            start_time = time.time()
            print("start_time: {}\nconfig -- lr:{} weight_decay:{} momentum:{} batch_size:{} epoches:{}".format(start_time, base_lr, weight_decay, momentum, batch_size, epoches))
            
            epoch, i, iterations_per_epoch_train = 0.0, 0, self.data.get_data_len()//batch_size
            while epoch < epoches:
                if i == 0: self.sess.run(tf.assign(self.net["lr"],base_lr))
                if i == 10*iterations_per_epoch_train:
                    new_lr = 1e-4
                    self.saver["lr"].save(self.sess, os.path.join(self.config.get("saver_path",SAVER_PATH),"lr-%f"%base_lr), global_step=i)
                    self.sess.run(tf.assign(self.net["lr"], new_lr))
                    base_lr = new_lr
                if i == 20*iterations_per_epoch_train:
                    new_lr = 1e-5
                    self.saver["lr"].save(self.sess, os.path.join(self.config.get("saver_path",SAVER_PATH),"lr-%f"%base_lr), global_step=i)
                    self.sess.run(tf.assign(self.net["lr"],new_lr))
                    base_lr = new_lr
                data_x, data_y, data_c, data_id_of_image = self.sess.run([x, y, c, id_of_image])
                params = {self.net["input"]:data_x, self.net["label"]:data_y, self.net["cues"]:data_c, self.net["drop_prob"]:0.5}
                self.sess.run(self.net["accum_gradient_accum"], feed_dict=params)
                if i % self.accum_num == self.accum_num-1:
                    _, _ = self.sess.run(self.net["accum_gradient_update"]), self.sess.run(self.net["accum_gradient_clean"])
                if i%500 == 0:
                    loss_cl, loss_am, loss_l2, loss_total, lr = self.sess.run([self.loss["loss_cl"], self.loss["loss_am"], self.loss["l2"], self.loss["total"], self.net["lr"]], feed_dict=params)
                    print("{:.1f}th epoch, {}iters, lr={:.5f}, loss={:.5f}+{:.5f}+{:.5f}={:.5f}".format(epoch, i, lr, loss_cl, loss_am, weight_decay*loss_l2, loss_total))
                if i%3000 == 2999:
                    self.saver["norm"].save(self.sess, os.path.join(self.config.get("saver_path",SAVER_PATH),"norm"), global_step=i)
                i+=1
                epoch = i/iterations_per_epoch_train
            end_time = time.time()
            print("end_time:{}\nduration time:{}".format(end_time, (end_time-start_time)))


if __name__ == "__main__":
    opt = parse_arg()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    batch_size = 1 # actual batch size=batch_size*accum_num
    input_size, category_num, epoches = (321,321), 21, 30
    data = dataset({"batch_size":batch_size, "input_size":input_size, "epoches":epoches, "category_num":category_num, "categorys":["train"]})
    if opt.restore_iter_id == None: gain = GAIN({"data":data, "batch_size":batch_size, "input_size":input_size, "epoches":epoches, "category_num":category_num, "init_model_path":"./model/init.npy", "accum_num":12})
    else: gain = GAIN({"data":data, "batch_size":batch_size, "input_size":input_size, "epoches":epoches, "category_num":category_num, "model_path":"{}/norm-{}".format(SAVER_PATH, opt.restore_iter_id), "accum_num":12})
    gain.train(base_lr=1e-3, weight_decay=5e-5, momentum=0.9, batch_size=batch_size, epoches=epoches, gpu_frac=float(opt.gpu_frac))