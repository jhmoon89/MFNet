import os, cv2
import numpy as np
import sys
from PIL import Image
from xml.etree.cElementTree import parse
import xml.etree.ElementTree as ET
# import dataset_utils.datasetUtils as datasetUtils
import time

#################################################
# import dataset_utils.datasetUtils as datasetUtils
#################################################
# If the script importing the module is in a package
# from .. import datasetUtils
#################################################
# If the script importing the module is not in a package
import inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import datasetUtils
#################################################
from pathos.multiprocessing import ProcessingPool as Pool


# from multiprocessing import Pool
#
# pool = Pool(processes=8)
def imageResize(imagePath, imageSize, bbox):
    image = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    if bbox != None:
        imageBbox = image[bbox[2]:bbox[3], bbox[0]:bbox[1], :]
        if len(imageBbox) == 0 or len(imageBbox[0]) == 0:
            imageResult = image
        else:
            imageResult = imageBbox
    else:
        imageResult = image
    imageResult = datasetUtils.imgAug(imageResult)
    imageResult = cv2.resize(imageResult, imageSize)
    return imageResult


class imagenetVidDataset(object):
    def __init__(self, dataPath, classNum=31, concurrentLength=4):
        self._classes = ['__background__',  # always index 0
                        'airplane', 'antelope', 'bear', 'bicycle',
                        'bird', 'bus', 'car', 'cattle',
                        'dog', 'domestic_cat', 'elephant', 'fox',
                        'giant_panda', 'hamster', 'horse', 'lion',
                        'lizard', 'monkey', 'motorcycle', 'rabbit',
                        'red_panda', 'sheep', 'snake', 'squirrel',
                        'tiger', 'train', 'turtle', 'watercraft',
                        'whale', 'zebra']
        self._classes_map = ['__background__',  # always index 0
                            'n02691156', 'n02419796', 'n02131653', 'n02834778',
                            'n01503061', 'n02924116', 'n02958343', 'n02402425',
                            'n02084071', 'n02121808', 'n02503517', 'n02118333',
                            'n02510455', 'n02342885', 'n02374451', 'n02129165',
                            'n01674464', 'n02484322', 'n03790512', 'n02324045',
                            'n02509815', 'n02411705', 'n01726692', 'n02355227',
                            'n02129604', 'n04468005', 'n01662784', 'n04530566',
                            'n02062744', 'n02391049']
        self._dataPath = dataPath
        self._classNum = classNum
        self._epoch = 0
        self._dataStart = 0
        self._dataLength = 0
        self._dataPointPathList = None
        self._classIdxConverter = None
        self._imageSize = (416, 416)
        self._concurrentLength = concurrentLength
        self._cellNum = (int(self._imageSize[0] / 32), int(self._imageSize[1] / 32))
        self._B = 2
        self._outputDim = self._B * 5 + self._classNum
        # self._loadDataPointPath2()
        self._loadDataPointPath()
        self._dataShuffle()

    def setImageSize(self, size=(416, 416)):
        self._imageSize = (size[0], size[1])

    def _loadDataPointPath(self):
        print 'load data point path...'
        self._dataPointPathList = []
        self._classIdxConverter = dict()
#         with open(self._dataPath + "/class.txt") as textFile:
#             lines = [line.split(" ") for line in textFile]
#         print lines

        trainPath = os.path.join(self._dataPath, 'Data')
        trainPath = os.path.join(trainPath, 'VID')
        trainPath = os.path.join(trainPath, 'train')
        subtrainPathList = os.listdir(trainPath)
        subtrainPathList.sort(key=datasetUtils.natural_keys)
        # print subtrainPathList
        subsubtrainPathList = []
        self._dataPointPathList = []
        for subtrainpath in subtrainPathList:
            sub_sub_train_path = os.listdir(os.path.join(trainPath, subtrainpath))
            sub_sub_train_path.sort(key=datasetUtils.natural_keys)
            subsubtrainPathList.append(sub_sub_train_path)
            for k in sub_sub_train_path:
                semi_finalPath = os.path.join(trainPath, subtrainpath, k)
                imgPathList = os.listdir(semi_finalPath)
                imgPathList.sort(key=datasetUtils.natural_keys)
                if len(imgPathList) < self._concurrentLength:
                    continue
                for img in range(len(imgPathList)-self._concurrentLength+1):
                    finalPath = os.path.join(semi_finalPath, imgPathList[img])
                    self._dataPointPathList.append(finalPath)
#         print self._dataPointPathList[-1]
#         print subsubtrainPathList[0]
#         print len(subsubtrainPathList)
#         print len(subsubtrainPathList[3])
        # print len(self._dataPointPathList)
        self._dataLength = len(self._dataPointPathList)
        print 'load done!'

    def _dataShuffle(self):
        # 'data list shuffle...'
        self._dataStart = 0
        np.random.shuffle(self._dataPointPathList)
        print "shuffle done!\n"
        # print len(self._dataPointPathList)
        # print self._dataPointPathList[0]

    def showImage(self, path, ind):
        index = str(ind).zfill(6)
        img_name = path + '/' + index + '.JPEG'

        # print img_name
        img = cv2.imread(img_name)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def drawRect(self, img, box):
        img = cv2.rectangle(img, (box[1], box[3]), (box[0], box[2]), (255, 0, 0), 3)
        return img

    def getConcurrentImages(self, path, startind):
        seq_length = len(os.listdir(path))
        #if startind + self._concurrentLength -1 > seq_length -1:
        #    print 'Concurrent image frames should be inside the sequence!'
        #    return
        img_List = np.zeros([self._imageSize[0], self._imageSize[1], 3 * self._concurrentLength])

        output_box = np.zeros([self._cellNum[0], self._cellNum[1], 4 * self._B], np.float32) ### 13 by 13 by 4B
        output_conf = np.zeros([self._cellNum[0], self._cellNum[1], self._B], np.float32)  ### 13 by 13 by B
        output_cls = np.zeros([self._cellNum[0], self._cellNum[1], self._classNum], np.float32)  ### 13 by 13 by classNum
        final_output = np.zeros([self._cellNum[0], self._cellNum[1], self._outputDim], np.float32)

        # For loop for getting concurrent images
        for i in range(self._concurrentLength):
            index = str(startind + i).zfill(6)
            img_name = path + '/' + index + '.JPEG'

            img = cv2.imread(img_name, cv2.IMREAD_COLOR).astype('float')
            img = cv2.resize(img, self._imageSize)

            img_List[:,:,i * 3:(i+1) * 3] = img

            if i < self._concurrentLength - 1:
                continue

            ### Read xml for only last frame
            xml_path = str.replace(img_name, 'Data', 'Annotations')
            xml_path = str.replace(xml_path, 'JPEG', 'xml')

            # print img_name
            # print xml_path

            startTime = time.time()
            xml_file = parse(xml_path)
            note = xml_file.getroot()
            obj_list_temp = note.findall('object')
            img_size = [float(note.find('size').findtext('width')), float(note.find('size').findtext('height'))]
            img_resize_scale = [float(self._imageSize[0] / img_size[0]), float(self._imageSize[1] / img_size[1])]

            if obj_list_temp == []: ### skip to next loop if there is no object on the image
                continue

            obj_num = len(obj_list_temp)

            for j in range(obj_num):

                obj_temp = obj_list_temp[j]
                xmax = int(obj_temp.find('bndbox').findtext('xmax'))
                xmin = int(obj_temp.find('bndbox').findtext('xmin'))
                ymax = int(obj_temp.find('bndbox').findtext('ymax'))
                ymin = int(obj_temp.find('bndbox').findtext('ymin'))
                trackid = int(obj_temp.findtext('trackid'))
                obj_ind = int(self._classes_map.index(obj_temp.findtext('name')))
                # print obj_ind
                obj_class = self._classes[obj_ind]
                # print obj_class

                xc = ((xmax + xmin) / 2.0) * img_resize_scale[0]
                yc = ((ymax + ymin) / 2.0) * img_resize_scale[1]
                nx = int(xc / float(self._imageSize[0] / self._cellNum[0]))
                ny = int(yc / float(self._imageSize[1] / self._cellNum[1]))

                x_offset = (xc - nx * float(self._imageSize[0] / self._cellNum[0])) / float(self._imageSize[0] / self._cellNum[0])
                y_offset = (yc - ny * float(self._imageSize[1] / self._cellNum[1])) / float(self._imageSize[1] / self._cellNum[1])

                # print "before"
                # print (ny, nx)

                conf_nonzero_ind = (output_conf[ny, nx, :]==0).argmax(axis=-1)
                if conf_nonzero_ind >= self._B:
                    continue

                output_conf[ny, nx, conf_nonzero_ind] = 1
                box = np.array([x_offset
                               , y_offset
                               , float(xmax - xmin) / img_size[0]
                               , float(ymax - ymin) / img_size[1]
                                ]
                               , np.float32)
                # print box
                output_box[ny, nx, conf_nonzero_ind * 4:(conf_nonzero_ind + 1)*4] = box
                output_cls[ny, nx, obj_ind] = 1
                # print output_cls[ny, nx, :]

            final_output = np.concatenate([output_box, output_conf, output_cls], -1)

        #     ############## Let's check if output numpy(tensor) is assigned well ##############
        #     #### Read all Bounding Boxes on each frame ####
        #     img = cv2.imread(img_name)
        #
        #     obj_exist_ind = np.where(output_conf==1)
        #     print obj_exist_ind
        #     obj_exist_mat = np.array(obj_exist_ind)
        #
        #     for i in range(obj_num):
        #         ny_temp = obj_exist_ind[0][i]
        #         nx_temp = obj_exist_ind[1][i]
        #
        #         # print "after"
        #         # print (ny_temp, nx_temp)
        #         offset_x = output_box[ny_temp, nx_temp, 0]
        #         offset_y = output_box[ny_temp, nx_temp, 1]
        #         w = output_box[ny_temp, nx_temp, 2]
        #         h = output_box[ny_temp, nx_temp, 3]
        #
        #         c_x = (nx_temp + offset_x) * 32
        #         c_y = (ny_temp + offset_y) * 32
        #
        #         W_scale = w * img_size[0]
        #         H_scale = h * img_size[1]
        #
        #         c_x_scale = c_x / img_resize_scale[0]
        #         c_y_scale = c_y / img_resize_scale[1]
        #
        #         xmax_temp = int(c_x_scale + W_scale / 2.0)
        #         xmin_temp = int(c_x_scale - W_scale / 2.0)
        #         ymax_temp = int(c_y_scale + H_scale / 2.0)
        #         ymin_temp = int(c_y_scale - H_scale / 2.0)
        #
        #         img = self.drawRect(img, [xmax_temp, xmin_temp, ymax_temp, ymin_temp])
        #
        #     print ("object number is {:d}".format(obj_num))
        #     # print output_conf[..., 0]
        #     # print output_conf[..., 1]
        #
        # ### show Image ###
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return np.array(img_List), final_output

    def newEpoch(self):
        self._epoch += 1
        self._dataStart = 0
        self._dataShuffle()

    def getNextBatch(self, batchSize=32):
        verystartTime = time.time()
        if self._dataStart + batchSize >= self._dataLength:
            self.newEpoch()
        dataStart = self._dataStart
        dataEnd = dataStart + batchSize
        self._dataStart = self._dataStart + batchSize

        # print "batch epoch is {:f}".format(self._epoch)

        # Getting Batch
        dataPathTemp = self._dataPointPathList[dataStart:dataEnd]
        # print dataPathTemp
        ImgBatchData = []
        OutBatchData = []
        # xml_path_list = []

        #         print ("after batch selection: {0:.6f}".format(time.time()-startTime))

        # Batch is composed of batchsize * concurrentLength number of image frames
        for path in dataPathTemp:
            box_list = []
            startTime = time.time()
            
            parent_path = os.path.abspath(os.path.join(path, '..')) # get the parent path
            ind = int(path.split('/')[-1].replace('.JPEG','')) # get the image index
            
#             print parent_path
#             print ind

            # then obtain concurrent images
            frame_list, LastFrame_Output = self.getConcurrentImages(parent_path, ind)
#             print frame_list
#             print LastFrame_Output

            ImgBatchData.append(frame_list)
            OutBatchData.append(LastFrame_Output)
        # print ("append: {0:.6f}".format(time.time()-startTime))

        ######## Display Image ########
        # self.showImage(path, ind)


        #         startTime = time.time()
        final_batchData = {
            'Paths': dataPathTemp,
            'Images': np.array(ImgBatchData),
            'Outputs': np.array(OutBatchData)
        }

        consumedTime = time.time() - verystartTime
        # print("consumed time: {0:.3f}".format(consumedTime))

        return final_batchData



################### test ##################

# # data_path = '/hdd/data/ILSVRC/Data/VID'
# # data_path = '/ssdubuntu/data/ILSVRC/Data/VID'


# sample_img_path = '/ssdubuntu/data/ILSVRC/Data/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000'
# sample_xml_path = '/ssdubuntu/data/ILSVRC/Annotations/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00008006/000000.xml'
#
# # vid_data.showImage(sample_img_path, 0)

# # sample = vid_data.getNextBatch()
# # print(sample.get('Boxes'))
# # print(type(sample.get('Images')))
# # print(sample.get('Paths'))
# print np.shape(sample.get('Images'))
# print np.shape(sample.get('Outputs'))
# # vid_data.showImage(sample.get('Paths')[1], 0)

#################################################################
# data_path = '/ssdubuntu/data/ILSVRC'
# vid_data = imagenetVidDataset(data_path)
# sample = vid_data.getNextBatch(4)
# # print sample.get('Paths')
# # print vid_data._dataPointPathList
# # #
# boxes = sample.get('Outputs')
# print np.shape(boxes)

# images = sample.get('Images')
# print np.shape(images)

# print images[0, 0, 0, :]
# first_conf = boxes[..., 4]
# ind = np.argwhere(first_conf==1)
# print ind[0]
#
# print boxes[ind[0][0], ind[0][1], ind[0][2], 0:4]