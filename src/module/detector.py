import numpy as np
import tensorflow as tf
import cv2
import time
import os
import sys


########################################################
# import inspect

# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(current_dir)
# sys.path.insert(0, parent_dir)

#import net_core.mfnet as mfnet
########################################################
import src.net_core.mfnet as mfnet


class mfnet_detector(object):
    def __init__(self,
                 dataPath='./',
                 nameScope='MF',
                 imgSize=(416, 416),
                 batchSize=32,
                 learningRate=0.0001,
                 classNum=31,
                 coreActivation=tf.nn.relu,
                 lastActivation=tf.nn.softmax,
                 concurrentFrame=4,
                 bboxNum=2
                 ):
        self._imgList = None
        self._imgClassList = None
        self._dataPath = dataPath
        self._nameScope = nameScope
        self._imgSize = imgSize
        self._batchSize = batchSize
        self._lr = learningRate
        self._coreAct = coreActivation
        self._lastAct = lastActivation
        self._classNum = classNum
        self.variables = None
        self.update_ops = None
        self._inputImgs = None
        #         self._inputImgs = tf.placeholder(tf.float32, shape=(None, 416, 416, 30))
        self._output = None
        self._outputGT = None
        self._optimizer = None
        self._loss = None
        self._concurrentFrame = concurrentFrame
        self._bboxNum = bboxNum
        self._lambdacoord = 5.0
        self._lambdaobj = 1.0
        self._lambdanoobj = 0.5
        self._lambdacls = 1.0
        self._objnum = 10  ### number of maximum object detected in a single frame
        self._indicators = None
        self._classes = None
        self._detTh = 0.5

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)
        # self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


        # initialize Vars

        self._buildNetwork()

        # self._sess.close()

        self._createLoss()
        self._setOptimizer()
        self._createEvaluation()

        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        self._sess = tf.Session()
        self._sess.run(init)

        # self._sess.close()

    def _buildNetwork(self):
        print "build Network..."
        #######################################################
        tf.reset_default_graph()  ### modified for ipython!!!##
        #######################################################
        self._inputImgs = tf.placeholder(tf.float32, shape=(None, self._imgSize[0],
                                                            self._imgSize[1], 3 * self._concurrentFrame))
        # self._inputImgs = tf.reshape(self._inputImgs, [-1, self._imgSize[0],
        #                                                self._imgSize[1], 3 * self._concurrentFrame])

        outputDim = self._bboxNum * 5 + self._classNum
        self._outputGT = tf.placeholder(tf.float32,
                                        shape=(None, int(self._imgSize[0] / 32),
                                               int(self._imgSize[1] / 32), outputDim))
        # self._outputGT = tf.reshape(self._outputGT, [-1, int(self._imgSize[0] / 32),
        #                                              int(self._imgSize[1] / 32), outputDim])
        # print self._outputGT.get_shape()

        self._detector = mfnet.MF_Detection(outputDim=outputDim,
                                            nameScope=self._nameScope + 'detector',
                                            trainable=True,
                                            bnPhase=True,
                                            reuse=False,  ######## here modified for ipython(True), normally False!!
                                            coreActivation=self._coreAct,
                                            lastLayerActivation=self._lastAct
                                            )
        #         self._output = self._detector(self._inputImgs)
        self._output = self._detector(self._inputImgs)
        # print self._output.get_shape()
        print "build Done!"
        # self._sess.close()

    # def get_iou(self, box1, box2):
    #     # transform (x, y, w, h) to (x1, y1, x2, y2)
    #     box1_t = np.array([box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]])
    #     box2_t = np.array([box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]])
    #
    #     # left upper and right down
    #     lu = np.array([np.maximum(box1_t[0], box2_t[0]), np.maximum(box1_t[1], box2_t[1])])
    #     rd = np.array([np.minimum(box1_t[2], box2_t[2]), np.minimum(box1_t[3], box2_t[3])])
    #
    #     # print lu, rd
    #     # intersection
    #     intersection = np.maximum(0.0, rd - lu)
    #     # print intersection
    #     inter_area = intersection[0] * intersection[1]
    #
    #     # calculate each box area
    #     box1_area = box1[2] * box1[3]
    #     box2_area = box2[2] * box2[3]
    #
    #     union = inter_area / (box1_area + box2_area - inter_area)
    #     return union
    #
    #
    # def find_max_iou_box(self, box, box_ref_list):
    #     iou_list = []
    #     for j in range(len(box_ref_list)):
    #         iou_list.append(self.get_iou(box, box_ref_list[j]))
    #     return box_ref_list[np.argmax(iou_list)]

    def duplicate_each_element(self, T, num):  # duplicate each element 'num' times for last dimension
        temp = tf.expand_dims(T, -1)
        shape_list = [1] * len(T.get_shape()) + [num]
        temp = tf.tile(temp, shape_list)
        T_shape = T.get_shape().as_list()

        temp = tf.reshape(temp, [-1] + T_shape[1:-1] + [T_shape[-1] * num])
        return temp

    def merge_last_two_dimensions(self, T):  ## T : (None, ....)
        T_shape = T.get_shape().as_list()
        new_shape = [-1] + T_shape[1:-2] + [T_shape[-2] * T_shape[-1]]
        temp = tf.reshape(T, new_shape)
        return temp
    
    def split_last_dimension(self, T, last_size): ## T : (None, ... (n * last_size) )
        T_shape = T.get_shape().as_list()
        assert T_shape[-1] % last_size == 0
        new_shape = [-1] + T_shape[1:-1] + [T_shape[-1]/last_size] + [last_size]
        temp = tf.reshape(T, new_shape) ## output : (None, ..., n, last_size)
        return temp
    
    def wh_sqrt(self, boxes): ## boxes : (None, 13, 13, bboxNum, 4)
        w = boxes[..., 2]
        h = boxes[..., 3]
        wh_sqrt = tf.stack([tf.sqrt(w), tf.sqrt(h)], -1)
        xy = boxes[..., :2]
        return tf.concat([xy, wh_sqrt], -1)
    
    def xy_add_offset(self, boxes): ## boxes : (None, 13, 13, bboxNum, 4)
        x = boxes[..., 0]
        y = boxes[..., 1]
        
        batchSize = tf.shape(self._output)[0]
        gridSize = [int(self._imgSize[0] / 32), int(self._imgSize[1] / 32)]
        offset_col = np.transpose(np.reshape(np.array(
            [np.arange(gridSize[0])] * gridSize[1] * self._bboxNum),
                                             (self._bboxNum, gridSize[0], gridSize[1])), (1, 2, 0))
        offset_row = np.transpose(offset_col, (1, 0, 2))
        
        offset_col_tf = tf.tile(tf.reshape(tf.constant(offset_col, dtype=tf.float32),
                                              [1, gridSize[0], gridSize[1], self._bboxNum]), [batchSize, 1, 1, 1])

        offset_row_tf = tf.tile(tf.reshape(tf.constant(offset_row, dtype=tf.float32),
                                              [1, gridSize[0], gridSize[1], self._bboxNum]), [batchSize, 1, 1, 1])
        
        x = x + offset_col_tf
        y = y + offset_row_tf
        
        x = tf.expand_dims(x, -1)
        y = tf.expand_dims(y, -1)
        
        xy = tf.concat([x, y], -1)
        wh = boxes[..., 2:]
        
        return tf.concat([xy, wh], -1)
        
    def calc_iou(self, boxes1, boxes2):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """

        # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
        boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                             boxes1[..., 1] - boxes1[..., 3] / 2.0,
                             boxes1[..., 0] + boxes1[..., 2] / 2.0,
                             boxes1[..., 1] + boxes1[..., 3] / 2.0],
                            axis=-1)

        boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                             boxes2[..., 1] - boxes2[..., 3] / 2.0,
                             boxes2[..., 0] + boxes2[..., 2] / 2.0,
                             boxes2[..., 1] + boxes2[..., 3] / 2.0],
                            axis=-1)

        # calculate the left up point & right down point
        lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
        rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

        # intersection
        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[..., 0] * intersection[..., 1]

        # calculate the boxs1 square and boxs2 square
        square1 = boxes1[..., 2] * boxes1[..., 3]
        square2 = boxes2[..., 2] * boxes2[..., 3]

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def eval_from_table(self, pred_list, gt_list):
        # pred_list : predicted object list. array of (x,y,w,h,conf)
        # gt_list : gt object list. array of (x,y,w,h,conf)

        TP = 0
        FP = 0
        FN = 0

        gt_num = len(gt_list)
        pred_num = len(pred_list)

        iou_mat = np.zeros([gt_num, pred_num], np.float32)

        for gt in range(gt_num):
            for pred in range(pred_num):
                iou_mat[gt, pred] = self.get_iou(gt_list[gt], pred_list[pred])
            if iou_mat[gt, :] > 0.5:
                TP = TP + 1
            else:
                FN = FN + 1

        for pred in range(pred_num):
            if iou_mat[:, pred] <= 0.5:
                FP = FP + 1

        return [TP, FP, FN]

    def _createLoss(self):
        print "create loss..."
        ################################################################
        ## pred
        boxes_pred = self._output[..., :4 * self._bboxNum]
        conf_pred = self._output[..., 4 * self._bboxNum : 5 * self._bboxNum]
        cls_pred = self._output[..., -1 * self._classNum:]
        
        ## get into resonable range
        boxes_pred = tf.nn.sigmoid(boxes_pred)
        conf_pred = tf.nn.sigmoid(conf_pred)
        cls_pred = tf.nn.softmax(cls_pred)
        
        ## reshape it
        boxes_pred = self.split_last_dimension(boxes_pred, 4) ## (None, 13, 13, bboxNum, 4)
        
        ## get [x, y, sqrt(w), sqrt(h)]
        boxes_sqrt_pred = self.wh_sqrt(boxes_pred) ## (None, 13, 13, bboxNum, 4)
        
               
        ## get [x + nx, y + ny, w, h]
        boxes_offset_pred = self.xy_add_offset(boxes_pred) ## (None, 13, 13, bboxNum, 4)

                
        #################################################################
        ## gt
        boxes_gt = self._outputGT[..., :4 * self._bboxNum]
        conf_gt = self._outputGT[..., 4 * self._bboxNum : 5 * self._bboxNum]
        cls_gt = self._outputGT[..., -1 * self._classNum:]
        
        ## reshape it
        boxes_gt = self.split_last_dimension(boxes_gt, 4) ## (None, 13, 13, bboxNum, 4)
        
        ## get [x, y, sqrt(w), sqrt(h)]
        boxes_sqrt_gt = self.wh_sqrt(boxes_gt) ## (None, 13, 13, bboxNum, 4)
        
        
        ## get [x + nx, y + ny, w, h]
        boxes_offset_gt = self.xy_add_offset(boxes_gt) ## (None, 13, 13, bboxNum, 4)
        
        #################################################################
        ## tile tensors
        
        #################################
        ## pred should be tiled as (1, 2, ..., n), (1, 2, ..., n)
        
        # tile prediction box
        tile_boxes_pred = tf.tile(boxes_offset_pred, [1, 1, 1, self._bboxNum, 1])
        # (None, 13, 13, bboxNum * bboxNum, 4)       
        
        # tile prediction confidence
        tile_conf_pred = tf.tile(conf_pred, [1, 1, 1, self._bboxNum])
        # (None, 13, 13, bboxNum * bboxNum)
        
        #################################
        ## gt should be tiled as (1, 1, 1, ... 1), (2, 2, ..., 2)
        
        # tile gt box(offset)
        tile_boxes_gt = tf.tile(boxes_offset_gt, [1, 1, 1, 1, self._bboxNum])
        shape_boxes_gt = tile_boxes_gt.get_shape().as_list()
        new_shape_boxes_gt = [-1] + shape_boxes_gt[1:3] + [self._bboxNum * self._bboxNum, 4]
        tile_boxes_gt = tf.reshape(tile_boxes_gt, new_shape_boxes_gt)
        # (None, 13, 13, bboxNum * bboxNum, 4)
        
        # tile gt box(sqrt)
        tile_boxes_sqrt_gt = tf.tile(boxes_sqrt_gt, [1, 1, 1, 1, self._bboxNum])
        
        # tile gt confidence
        tile_conf_gt = self.duplicate_each_element(conf_gt, self._bboxNum)
        # (None, 13, 13, bboxNum * bboxNum)   
        #################################
        #################################################################
        ## get iou for tiled boxes(offset added) and get mask
        
        iou = self.calc_iou(tile_boxes_pred, tile_boxes_gt)
        
        # split iou to get one-hot vector
        iou = self.split_last_dimension(iou, self._bboxNum)
        _, max_iou_indices = tf.nn.top_k(iou, 1)
        max_iou_indices = tf.squeeze(max_iou_indices, -1)
        mask_obj = tf.cast(tf.one_hot(max_iou_indices, depth=iou.get_shape()[-1]), tf.bool)
        
        mask_noobj = tf.cast(tf.logical_not(tf.reduce_any(mask_obj, -2)), tf.float32)        
        mask_obj = tf.cast(self.merge_last_two_dimensions(mask_obj), tf.float32)
        
#         print mask_obj.get_shape()
#         print tile_conf_gt.get_shape()
        obj_mat = tile_conf_gt * mask_obj ## (None, 13, 13, bboxNum * bboxNum)
        noobj_mat = mask_noobj ## (None, 13, 13, bboxNum)
        # noobj_mat indicates the cells which are not responsible for any of gt cells
        
        #################################################################
        ## obtaining losses
    
        # box loss
        obj_mat_box = self.duplicate_each_element(obj_mat, 4)
        # (None, 13, 13, 4 * bboxNum * bboxNum)
        tile_boxes_sqrt_pred = tf.tile(boxes_sqrt_pred, [1, 1, 1, self._bboxNum, 1])
        tile_boxes_sqrt_pred = self.merge_last_two_dimensions(tile_boxes_sqrt_pred) # (1, 2, ...)
        
        tile_boxes_sqrt_gt = tf.tile(boxes_sqrt_gt, [1, 1, 1, 1, self._bboxNum])
        tile_boxes_sqrt_gt = self.merge_last_two_dimensions(tile_boxes_sqrt_gt) # (1, 1, )
        
        box_loss = tf.squared_difference(tile_boxes_sqrt_gt, tile_boxes_sqrt_pred) * obj_mat_box
        box_loss = tf.reduce_sum(box_loss, -1)
        box_loss = tf.reduce_sum(box_loss, -1)
        box_loss = tf.reduce_sum(box_loss, -1)
        self._box_loss = tf.reduce_mean(box_loss)
        
        # obj loss
        conf_obj_loss = tf.squared_difference(tile_conf_gt, tile_conf_pred) * obj_mat
        conf_obj_loss = tf.reduce_sum(conf_obj_loss, -1)
        conf_obj_loss = tf.reduce_sum(conf_obj_loss, -1)
        conf_obj_loss = tf.reduce_sum(conf_obj_loss, -1)
        self._conf_obj_loss = tf.reduce_mean(conf_obj_loss)
        
        # no obj loss
        conf_noobj_loss = tf.square(conf_pred) * noobj_mat # conf_gt cell is 0
        conf_noobj_loss = tf.reduce_sum(conf_noobj_loss, -1)
        conf_noobj_loss = tf.reduce_sum(conf_noobj_loss, -1)
        conf_noobj_loss = tf.reduce_sum(conf_noobj_loss, -1)
        self._conf_noobj_loss = tf.reduce_mean(conf_noobj_loss)
        
        # class loss
        cls_mat = tf.cast(tf.reduce_any(tf.cast(conf_gt, tf.bool), -1), tf.float32)
        class_crossentropy = -1 * tf.reduce_sum(tf.multiply(cls_gt, tf.log(cls_pred + 1e-9)),-1)
        class_loss = class_crossentropy * cls_mat
        class_loss = tf.reduce_sum(class_loss, -1)
        class_loss = tf.reduce_sum(class_loss, -1)     
        self._class_loss = tf.reduce_mean(class_loss)
        
        # final loss
        self._loss = self._lambdacoord * self._box_loss \
                    + self._lambdaobj * self._conf_obj_loss \
                    + self._lambdanoobj * self._conf_noobj_loss \
                    + self._lambdacls * self._class_loss

        
        #################################################################

#         batchSize = tf.shape(self._output)[0]
#         gridSize = [int(self._imgSize[0] / 32), int(self._imgSize[1] / 32)]
#         offset_col = np.transpose(np.reshape(np.array(
#             [np.arange(gridSize[0])] * gridSize[1] * self._bboxNum),
#             (self._bboxNum, gridSize[0], gridSize[1])), (1, 2, 0))

#         offset_row = np.transpose(offset_col, (1, 0, 2))
#         offset_col_tf = tf.tile(tf.reshape(tf.constant(offset_col, dtype=tf.float32),
#                                               [1, gridSize[0], gridSize[1], self._bboxNum]), [batchSize, 1, 1, 1])

#         offset_row_tf = tf.tile(tf.reshape(tf.constant(offset_row, dtype=tf.float32),
#                                               [1, gridSize[0], gridSize[1], self._bboxNum]), [batchSize, 1, 1, 1])
        

###################################################################################################        
#         self._objMask = []
#         self._objMask_dup = []  ## Mask for just copying elements four times for boxloss
#         self._objMask_bool = tf.cast(
#             tf.tile(
#                 tf.expand_dims(
#                     tf.zeros(
#                         tf.shape(self._output[...,0])), -1),[1, 1, 1, self._bboxNum]), tf.bool)  ## Mask for inverting non-obj-mask


#         class_pred = tf.nn.softmax(self._output[..., -1 * self._classNum:])
#         boxes_pred = tf.sigmoid(self._output[..., :4 * self._bboxNum])
#         boxes_pred_reshape = tf.reshape(boxes_pred,
#                                         [-1, boxes_pred.get_shape()[1], boxes_pred.get_shape()[2], self._bboxNum, 4])
#         xy_pred = boxes_pred_reshape[..., :2]
#         x_pred = boxes_pred_reshape[..., 0]
#         y_pred = boxes_pred_reshape[..., 1]

#         wh_pred = boxes_pred_reshape[..., 2:]
#         wh_pred_sqrt = tf.sqrt(wh_pred)
#         boxes_with_sqrt_pred = tf.concat([xy_pred, wh_pred_sqrt], -1)  # (None, 13, 13, bboxNum, 4)
#         boxes_with_sqrt_pred = self.merge_last_two_dimensions(boxes_with_sqrt_pred)  ## (None, 13, 13, 4 * bboxNum)
#         ## (xywh1, xywh2, ... xywhB) = 4 * bboxNum
#         boxes_with_sqrt_pred_tile = tf.tile(boxes_with_sqrt_pred, [1, 1, 1, self._bboxNum])
#         ## (1,2,...B), (1,2,...B), ..., (1,2,...,B)


#         conf_pred = tf.sigmoid(self._output[..., 4 * self._bboxNum: 5 * self._bboxNum])  ## (None, 13, 13, bboxNum)
#         conf_pred_dup = tf.tile(conf_pred, [1, 1, 1, self._bboxNum])  ## (c1, c2, ...cB), (c1, c2, ..., cB), ...

#         #         print conf_pred_dup.get_shape()





#         class_gt = tf.nn.softmax(self._outputGT[..., -1 * self._classNum:])
#         boxes_gt = tf.sigmoid(self._outputGT[..., :4 * self._bboxNum])
#         boxes_gt_reshape = tf.reshape(boxes_gt,
#                                       [-1, boxes_gt.get_shape()[1], boxes_gt.get_shape()[2], self._bboxNum, 4])
#         xy_gt = boxes_gt_reshape[..., :2]
#         x_gt = boxes_gt_reshape[..., 0]
#         y_gt = boxes_gt_reshape[..., 1]
#         wh_gt = boxes_gt_reshape[..., 2:]
#         wh_gt_sqrt = tf.sqrt(wh_gt)
#         boxes_with_sqrt_gt = tf.concat([xy_gt, wh_gt_sqrt], -1)  # (None, 13, 13, bboxNum, 4)
#         boxes_with_sqrt_gt_tile = tf.tile(boxes_with_sqrt_gt,
#                                           [1, 1, 1, 1, self._bboxNum])  # (None, 13, 13, bboxNum, 4 * bboxNum)
#         boxes_with_sqrt_gt_tile = self.merge_last_two_dimensions(boxes_with_sqrt_gt_tile)
#         ## (xywh1, xywh1, ...) = 4 * bboxNum
#         ## (1, 1, ..), (2, 2, ...), ... , (B, B, .., B)
#         boxes_with_sqrt_gt = self.merge_last_two_dimensions(boxes_with_sqrt_gt)  ## (None, 13, 13, 4 * bboxNum)

#         conf_gt = self._outputGT[..., 4 * self._bboxNum: 5 * self._bboxNum]  ## (None, 13, 13, bboxNum)
#         conf_gt_dup = self.duplicate_each_element(conf_gt, self._bboxNum)  ## (c1, c1, ...), (c2, c2, ...)
#         conf_gt_dup_box = self.duplicate_each_element(conf_gt,
#                                                       4 * self._bboxNum)  ## (None, 13, 13, 4 * bboxNum * bboxNum

#         #         print conf_gt_dup.get_shape()

#         ############## offset #############

#         batchSize = tf.shape(self._output)[0]
#         gridSize = [int(self._imgSize[0] / 32), int(self._imgSize[1] / 32)]
#         offset_col = np.transpose(np.reshape(np.array(
#             [np.arange(gridSize[0])] * gridSize[1] * self._bboxNum),
#             (self._bboxNum, gridSize[0], gridSize[1])), (1, 2, 0))

#         offset_row = np.transpose(offset_col, (1, 0, 2))
#         #         print np.shape(offset)

#         self._offset_col = tf.tile(tf.reshape(tf.constant(offset_col, dtype=tf.float32),
#                                               [1, gridSize[0], gridSize[1], self._bboxNum]), [batchSize, 1, 1, 1])

#         self._offset_row = tf.tile(tf.reshape(tf.constant(offset_row, dtype=tf.float32),
#                                               [1, gridSize[0], gridSize[1], self._bboxNum]), [batchSize, 1, 1, 1])

#         x_pred_offset = x_pred + self._offset_col
#         x_gt_offset = x_gt + self._offset_col
#         y_pred_offset = y_pred + self._offset_row
#         y_gt_offset = y_gt + self._offset_row

#         xy_pred_offset = tf.stack([x_pred_offset, y_pred_offset], -1)
#         boxes_pred_offset = tf.concat([xy_pred_offset, wh_pred], -1)

#         xy_gt_offset = tf.stack([x_gt_offset, y_gt_offset], -1)
#         boxes_gt_offset = tf.concat([xy_gt_offset, wh_gt], -1)
#         boxes_gt_offset = self.merge_last_two_dimensions(boxes_gt_offset)

#         self._boxes_pred_offset = boxes_pred_offset ### (None, 13, 13, bboxNum, 4)
#         self._boxes_gt_offset = tf.reshape(boxes_gt_offset, [-1, boxes_gt_offset.get_shape()[1], boxes_gt_offset.get_shape()[2],
#                                                              self._bboxNum, 4]) ### (None, 13, 13, bboxNum, 4)


#         ##### box loss #####

#         for j in range(self._bboxNum):
#             boxes_gt_tile = tf.tile(boxes_gt_offset[..., 4 * j:4 * (j + 1)], [1, 1, 1, self._bboxNum])
#             boxes_gt_tile_reshape = tf.reshape(boxes_gt_tile,
#                                                [-1, boxes_pred.get_shape()[1], boxes_pred.get_shape()[2], self._bboxNum,
#                                                 4])

#             iou = self.calc_iou(boxes_gt_tile_reshape, boxes_pred_offset)
#             iou_max, indices = tf.nn.top_k(iou)
#             indices = tf.squeeze(indices, -1)
#             mask_per_one_gt_box = tf.one_hot(indices=indices, depth=iou.get_shape()[-1])
#             mask_per_one_gt_box_dup = self.duplicate_each_element(mask_per_one_gt_box, 4)  # (None, 13, 13, 4*bboxNum)
#             #             print mask_per_one_gt_box.get_shape()

#             mask_bool = tf.cast(mask_per_one_gt_box, tf.bool)

#             self._objMask += [mask_per_one_gt_box]
#             self._objMask_dup += [mask_per_one_gt_box_dup]
#             self._objMask_bool = tf.logical_or(self._objMask_bool, mask_bool)

#         self._objMask = tf.concat(self._objMask, axis=-1)
#         self._objMask_dup = tf.concat(self._objMask_dup, axis=-1)
#         self._objMask_no_obj = tf.cast(tf.tile(tf.logical_not(self._objMask_bool), [1, 1, 1, self._bboxNum]), tf.float32)

#         box_diff_square = tf.squared_difference(boxes_with_sqrt_pred_tile, boxes_with_sqrt_gt_tile)
#         box_loss = tf.multiply(box_diff_square, self._objMask_dup)
#         box_loss = tf.multiply(box_loss, conf_gt_dup_box)
#         box_loss = tf.reduce_sum(box_loss, axis=-1)
#         box_loss = tf.reduce_sum(box_loss, axis=-1)
#         box_loss = tf.reduce_sum(box_loss, axis=-1)

#         self._box_loss = tf.reduce_mean(box_loss)

#         ##### conf loss #####
#         conf_diff_square = tf.squared_difference(conf_pred_dup, conf_gt_dup)
#         conf_obj_loss = tf.multiply(conf_diff_square, self._objMask)
#         #         conf_obj_loss = tf.multiply(conf_obj_loss, conf_gt_dup)
#         conf_obj_loss = tf.reduce_sum(conf_obj_loss, axis=-1)
#         conf_obj_loss = tf.reduce_sum(conf_obj_loss, axis=-1)
#         conf_obj_loss = tf.reduce_sum(conf_obj_loss, axis=-1)

#         self._conf_obj_loss = tf.reduce_mean(conf_obj_loss)

#         ##### conf loss for non object #####
#         no_obj_ind = self._objMask_no_obj
#         no_obj_conf = tf.subtract(tf.ones_like(conf_gt_dup), conf_gt_dup)
#         conf_noobj_loss = tf.multiply(conf_diff_square, no_obj_ind)
#         conf_noobj_loss = tf.multiply(conf_noobj_loss, no_obj_conf)
#         conf_noobj_loss = tf.reduce_sum(conf_noobj_loss, axis=-1)
#         conf_noobj_loss = tf.reduce_sum(conf_noobj_loss, axis=-1)
#         conf_noobj_loss = tf.reduce_sum(conf_noobj_loss, axis=-1)

#         self._conf_noobj_loss = tf.reduce_mean(conf_noobj_loss)

#         ##### class loss #####
#         obj_exist = conf_gt[..., 0]
#         #obj_exist = tf.expand_dims(obj_exist, -1)
#         #print obj_exist.get_shape()
#         #obj_exist = tf.tile(obj_exist, [1, 1, 1, self._classNum])

#         class_diff_square = tf.squared_difference(class_pred, class_gt)
#         class_crossentropy = -1 * tf.reduce_sum(tf.multiply(class_gt, tf.log(class_pred + 1e-9)),-1)
#         #print class_crossentropy.get_shape()
#         #class_loss = tf.multiply(obj_exist, class_diff_square)
#         class_loss = tf.multiply(obj_exist, class_crossentropy)
#         #print class_loss.get_shape()
#         class_loss = tf.reduce_sum(class_loss, -1)
#         #print class_loss.get_shape()
#         class_loss = tf.reduce_sum(class_loss, -1)
#         #class_loss = tf.reduce_sum(class_loss, -1)

#         self._class_loss = tf.reduce_mean(class_loss)

#         ###########################################################################################


#         ### final loss ###
#         weighted_box_loss = tf.multiply(self._lambdacoord, self._box_loss)
#         weighted_conf_loss_obj = tf.multiply(self._lambdaobj, self._conf_obj_loss)
#         weighted_conf_loss_noobj = tf.multiply(self._lambdanoobj, self._conf_noobj_loss)
#         weighted_class_loss = tf.multiply(self._lambdacls, self._class_loss)

#         losses = tf.stack([weighted_box_loss, weighted_conf_loss_obj, weighted_conf_loss_noobj, weighted_class_loss])

#         self._loss = tf.reduce_sum(losses)
#         # self._loss = tf.reduce_sum(tf.stack([tf.scalar_mul(self._lambdacoord, self._box_loss),
#         #                                      self._conf_loss_obj,
#         #                                      tf.scalar_mul(self._lambdanoobj, self._conf_loss_noobj),
#         #                                      self._class_loss
#         #                                      ]))

        print "create Done!"

    def _setOptimizer(self):
        print "set optimizer..."
        self._lr = tf.placeholder(tf.float32, shape=[])
        # self._optimizer = tf.train.MomentumOptimizer(learning_rate=self._lr, momentum=0.90, use_nesterov=True)
        with tf.control_dependencies(self._detector.allUpdate_ops):
            # self._optimizer = self._optimizer.minimize(self._loss, var_list=self._detector.allVariables)
            self._optimizer = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(self._loss, None,
                                                                                      var_list=self._detector.allVariables)
        print "set Done!"

    #
    def _createEvaluation(self):
        print "evaluation..."

#         # print self._objMask.get_shape()
#         # print self._objMask_dup.get_shape()

        ####################### Get Boxes List from gt and pred tensor #######################

        class_pred = tf.nn.softmax(self._output[..., -1 * self._classNum:]) ### (None, 13, 13, clsNum)
        conf_pred = tf.sigmoid(self._output[..., 4 * self._bboxNum: 5 * self._bboxNum]) ### (None, 13, 13, bboxNum)

#         class_pred_tile = tf.tile(class_pred, [1, 1, 1, self._bboxNum]) ### (None, 13, 13, bboxNum * clsNum)
#         ### (0 1 2 ... 31), (0 1 2 ... 31)
#         conf_pred_tile = self.duplicate_each_element(conf_pred, self._classNum) ### (None, 13, 13, bboxNum * clsNum)
#         ### (0 0 0 0... 0), (1 1 1 ... 1)

#         class_cond_prob = tf.multiply(class_pred_tile, conf_pred_tile)
#         class_cond_prob = tf.reshape(class_cond_prob,
#                                      [-1,
#                                       class_cond_prob.get_shape()[1],
#                                       class_cond_prob.get_shape()[2],
#                                       self._bboxNum,
#                                       self._classNum])

#         self._indices_pred = tf.where(tf.greater(class_cond_prob, tf.sigmoid(tf.ones(tf.shape(class_cond_prob)) * self._detTh)))


#         self._boxes_list_pred = tf.gather_nd(self._boxes_pred_offset, self._indices_pred[...,:-1])
#         conf_list_pred = tf.gather_nd(conf_pred, self._indices_pred[...,:-1])
#         prob_list_pred = tf.gather_nd(class_pred, self._indices_pred[...,:-2])

#         self._cond_prob_list_pred = tf.multiply(tf.tile(tf.expand_dims(conf_list_pred, -1),[1, self._classNum]), prob_list_pred)

#         ###########################################


        class_gt = self._outputGT[..., -1 * self._classNum:]
        conf_gt = self._outputGT[..., 4 * self._bboxNum: 5 * self._bboxNum]


#         indices_gt = tf.where(tf.equal(conf_gt, tf.ones(tf.shape(conf_gt))))
#         self._indices_gt = indices_gt

#         self._boxes_list_gt = tf.gather_nd(self._boxes_gt_offset, indices_gt)
#         conf_list_gt = tf.gather_nd(conf_gt, indices_gt)
#         prob_list_gt = tf.gather_nd(class_gt, indices_gt[...,:-1])

#         self._cond_prob_list_gt = tf.multiply(tf.tile(tf.expand_dims(conf_list_gt, -1), [1, self._classNum]), prob_list_gt)

#         # print self._cond_prob_list_pred.get_shape()
#         # print self._cond_prob_list_gt.get_shape()


#         ######################################################################################

        # get classification accuracy
        conf_bool_mask = tf.cast(tf.cast(tf.reduce_sum(conf_gt, -1), tf.bool), tf.float32)
        class_label_pred = tf.argmax(class_pred, -1)
        class_label_gt = tf.argmax(class_gt, -1)

        class_equality = tf.cast(tf.equal(class_label_gt, class_label_pred), tf.float32)
        class_equality_obj = tf.multiply(class_equality, conf_bool_mask)
        class_equality_obj = tf.reduce_sum(class_equality_obj, -1)
        class_equality_obj = tf.reduce_sum(class_equality_obj, -1)
        
        #print class_equality_obj.get_shape()

        obj_num_in_batch = tf.reduce_sum(conf_bool_mask, -1)
        obj_num_in_batch = tf.reduce_sum(obj_num_in_batch, -1)

        self._obj_num_in_batch = tf.reduce_sum(obj_num_in_batch)
        
        EachAcc = class_equality_obj / (obj_num_in_batch + 1e-9)
        
        self._classAcc = tf.reduce_mean(EachAcc)

        print "eval Done!"

    def fit(self, batchDict):
        feed_dict = {
            self._inputImgs: batchDict['Images'],
            self._outputGT: batchDict['Outputs'],
            self._lr: batchDict['LearningRate']
        }

        # xy_gt_list = self._sess.run([self._xy_gt_list], feed_dict=feed_dict)
        # print xy_gt_list

        opt, lossResult, box_loss, conf_obj_loss, conf_noobj_loss, class_loss, classAcc, objNum = self._sess.run(
            [self._optimizer, self._loss, self._box_loss, self._conf_obj_loss, self._conf_noobj_loss, self._class_loss
             , self._classAcc, self._obj_num_in_batch
             ],
            feed_dict=feed_dict)

        # box_loss = self._sess.run([self._box_loss], feed_dict=feed_dict)

        # print (output_gt[np.argwhere(output_gt==1)[0],0:3])

        print ("box loss is {:f}".format(box_loss))
        print ("conf obj loss is {:f}".format(conf_obj_loss))
        print ("conf no obj loss is {:f}".format(conf_noobj_loss))
        print ("class loss is {:f}".format(class_loss))
        print ("final loss is {:f}".format(lossResult))
        print ("classification accuracy is {:f}%".format(100 * classAcc))
        print ("obj Num is {:d}".format(int(objNum)))

        return lossResult, classAcc

    def saveDetectorCore(self, savePath='./'):
        CorePath = os.path.join(savePath, self._nameScope + '_detectorCore.ckpt')
        self._detector.coreSaver.save(self._sess, CorePath)

    def saveDetectorLastLayer(self, savePath='./'):
        LastPath = os.path.join(savePath, self._nameScope + '_detectorLastLayer.ckpt')
        self._detector.detectorSaver.save(self._sess, LastPath)

    def saveNetworks(self, savePath='./'):
        self.saveDetectorCore(savePath)
        self.saveDetectorLastLayer(savePath)

    def restoreDetectorCore(self, restorePath='./'):
        CorePath = os.path.join(restorePath, self._nameScope + '_detectorCore.ckpt')
        self._detector.coreSaver.restore(self._sess, CorePath)

    def restoreDetectorLastLayer(self, restorePath='./'):
        LastPath = os.path.join(restorePath, self._nameScope + '_detectorLastLayer.ckpt')
        self._detector.detectorSaver.restore(self._sess, LastPath)

    def restoreNetworks(self, restorePath='./'):
        self.restoreDetectorCore(restorePath)
        self.restoreDetectorLastLayer(restorePath)

    def saveSecondCore(self, savePath='./'):
        SecondPath = os.path.join(savePath, self._nameScope + '_detectorCore2.ckpt')
        self._detector.secondSaver.save(self._sess, SecondPath)

    def restoreSecondCore(self, restorePath='./'):
        SecondPath = os.path.join(restorePath, self._nameScope + '_detectorCore2.ckpt')
        self._detector.secondSaver.restore(self._sess, SecondPath)


# sample = mfnet_detector()

# mfnet_detector._sess.close()
