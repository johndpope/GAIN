import os
import sys
import math
import random
import pickle
import skimage
import numpy as np
import tensorflow as tf
import skimage.io as imgio
from datetime import datetime
import skimage.transform as imgtf

class dataset():
    def __init__(self,config={}):
        self.config = config
        self.w, self.h = self.config.get("input_size",(321,321))
        self.categorys = self.config.get("categorys",["train"])
        self.category_num = self.config.get("category_num",21)
        self.main_path = self.config.get("main_path",os.path.join("data","VOCdevkit","VOC2012"))
        self.ignore_label = self.config.get("ignore_label",255)
        self.default_category = self.config.get("default_category",self.categorys[0])
        self.img_mean = np.ones((self.w,self.h,3))
        self.img_mean[:,:,0] *= 104.00698793
        self.img_mean[:,:,1] *= 116.66876762
        self.img_mean[:,:,2] *= 122.67891434
        self.data_f,self.data_len = self.get_data_f()

    def get_data_len(self,category=None):
        return self.data_len[category if category is not None else self.default_category]

    def get_data_f(self):
        self.cues_data = pickle.load(open("data/localization_cues.pickle","rb"),encoding="iso-8859-1")
        data_f, data_len = {}, {}
        for category in self.categorys:
            data_f[category] = {"img":[],"gt":[],"label":[],"id":[],"id_for_slice":[]}
            data_len[category] = 0
        for one in self.categorys:
            if "train" in one:
                with open(os.path.join("data","train_id.txt"),"r") as f:
                    for id_identy, line in enumerate(f.readlines()):
                        id_name = line.strip("\n")
                        data_f[one]["id"].append(id_name)
                        data_f[one]["id_for_slice"].append(str(id_identy))
                        data_f[one]["img"].append(os.path.join(self.main_path,"JPEGImages","%s.jpg" % id_name))
                        data_f[one]["gt"].append(os.path.join(self.main_path,"SegmentationClassAug","%s.png" % id_name))
                    if "length" in self.config:
                        length = self.config["length"]
                        data_f[one]["id"] = data_f[one]["id"][:length]
                        data_f[one]["id_for_slice"] = data_f[one]["id_for_slice"][:length]
                        data_f[one]["img"] = data_f[one]["img"][:length]
                        data_f[one]["gt"] = data_f[one]["gt"][:length]
                        print("id:%s" % str(data_f[one]["id"]))
                        print("img:%s" % str(data_f[one]["img"]))
                        print("id_for_slice:%s" % str(data_f[one]["id_for_slice"]))
                data_len[one] = len(data_f[one]["id"])
            if "val" in one:
                with open(os.path.join("data","val_id.txt"),"r") as f:
                    for id_identy, line in enumerate(f.readlines()):
                        id_name = line.strip("\n")
                        data_f[one]["id"].append(id_name)
                        data_f[one]["id_for_slice"].append(str(id_identy))
                        data_f[one]["img"].append(os.path.join(self.main_path,"JPEGImages","%s.jpg" % id_name))
                        data_f[one]["gt"].append(os.path.join(self.main_path,"SegmentationClass","%s.png" % id_name))
                    if "length" in self.config:
                        length = self.config["length"]
                        data_f[one]["id"] = data_f[one]["id"][:length]
                        data_f[one]["id_for_slice"] = data_f[one]["id_for_slice"][:length]
                        data_f[one]["img"] = data_f[one]["img"][:length]
                        data_f[one]["gt"] = data_f[one]["gt"][:length]
                        print("id:%s" % str(data_f[one]["id"]))
                        print("img:%s" % str(data_f[one]["img"]))
                        print("id_for_slice:%s" % str(data_f[one]["id_for_slice"]))
                data_len[one] = len(data_f[one]["id"])
        print("len:%s" % str(data_len))
        return data_f,data_len

    def next_batch(self,category=None,batch_size=None,epoches=-1):
        category = self.default_category if category is None else category
        batch_size = self.config.get("batch_size",1) if batch_size is None else batch_size
        dataset = tf.data.Dataset.from_tensor_slices({"id":self.data_f[category]["id"], "id_for_slice":self.data_f[category]["id_for_slice"], "img_f":self.data_f[category]["img"], "gt_f":self.data_f[category]["gt"]})
        def m(x):
            img, gt = self.image_preprocess(tf.image.decode_image(tf.read_file(x["img_f"])), tf.image.decode_image(tf.read_file(x["gt_f"])), random_scale=False, flip=False, rotate=False)
            #img = self.image_preprocess(img,random_scale=True,flip=True,rotate=False)
            img, gt = tf.reshape(img,[self.h,self.w,3]), tf.reshape(gt,[self.h,self.w,1])
            def get_data(identy):
                identy, label, cues = identy.decode(), np.zeros([self.category_num]), np.zeros([41,41,21])
                label[self.cues_data["%s_labels"%identy]] = 1.0
                cues_i = self.cues_data["%s_cues"%identy]
                cues[cues_i[1], cues_i[2], cues_i[0]] = 1.0
                return label.astype(np.float32),cues.astype(np.float32)
            label, cues = tf.py_func(get_data, [x["id_for_slice"]], [tf.float32,tf.float32])
            label.set_shape([21])
            cues.set_shape([41,41,21])
            return img, gt, label, cues, x["id"]
        iterator = dataset.repeat(epoches).shuffle(self.data_len[category]).map(m).batch(batch_size).make_initializable_iterator()
        img, gt, label, cues, id_ = iterator.get_next()
        return img, gt, label, cues, id_, iterator

    def image_preprocess(self,img,gt,random_scale=True,flip=False,rotate=False):
        img = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(img, axis=0),(self.h, self.w)), axis=0)
        gt = tf.squeeze(tf.image.resize_nearest_neighbor(tf.expand_dims(gt, axis=0),(self.h, self.w)), axis=0)
        r,g,b = tf.split(axis=2,num_or_size_splits=3,value=img)
        img = tf.cast(tf.concat([b,g,r], 2), dtype=tf.float32)
        img -= self.img_mean
        return img, gt
