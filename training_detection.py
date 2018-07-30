import numpy as np
import time, sys
import copy
import tensorflow as tf
import dataset_utils.dataset_loader.ImagenetVid_dataset as ImagenetVid_dataset
import src.module.detector as detector
import gc
import math
import datetime

MFNetConfig = {
    'inputImgDim':[None, None, 3],
    'classDim':31
}

def trainMFNetDetector(
        MFNetConfig, batchSize=8, training_epoch=10,
        max_iter=10,
        learningRate = 0.001,
        savePath=None, restorePath=None
):
    datasetPath = '/ssdubuntu/data/ILSVRC'
    dataset = ImagenetVid_dataset.imagenetVidDataset(datasetPath)
    dataset.setImageSize((416,416))

    model = detector.mfnet_detector(classNum=MFNetConfig['classDim'])

    if restorePath != None:
        print 'restore weights...'
        model.restoreNetworks(restorePath)
        # model.restoreSecondCore(restorePath)


    loss = 0.0
    acc = 0.0
    epoch = 0
    iteration = 0
    run_time = 0.0
    if learningRate == None:
        learningRate == 0.001

    veryStart = time.time()
    print 'start training...'
    #
    # batchData = dataset.getNextBatch(batchSize=8)
    # batchData['LearningRate'] = learningRate
    # epochCurr = dataset._epoch
    # dataStart = dataset._dataStart
    # dataLength = dataset._dataLength
    #
    # loss = model.fit(batchData)


    while epoch < training_epoch:

        iteration = 0
        for cursor in range(max_iter):
            start = time.time()
            iteration = cursor
            batchData = dataset.getNextBatch(batchSize=batchSize)
            batchData['LearningRate'] = learningRate
            epochCurr = dataset._epoch
            dataStart = dataset._dataStart
            dataLength = dataset._dataLength
            if epochCurr != epoch:
                epoch = epochCurr
                break

            lossTemp, accTemp = model.fit(batchData)

            end = time.time()
            loss = float(loss * iteration + lossTemp) / float(iteration + 1.0)
            acc = float(acc * iteration + accTemp) / float(iteration + 1.0)
            run_time = (run_time * iteration + (end - start)) / float(iteration + 1.0)


            sys.stdout.write(
                "Epoch:{:03d} iter:{:05d} runtime:{:.3f} ".format(int(epoch + 1), int(iteration + 1), run_time))
            sys.stdout.write("cur/tot:{:07d}/{:07d} ".format(dataStart, dataLength))
            #sys.stdout.write("Current Loss={:.6f} ".format(lossTemp))
            sys.stdout.write("Average Loss={:.6f} ".format(loss))
            sys.stdout.write("Average Accuracy={:.6f}%".format(acc * 100))
            sys.stdout.write("\n")
            sys.stdout.flush()

            if math.isnan(loss):
                break

            if cursor != 0 and cursor % 2000 == 0:
                model.saveNetworks(savePath)

        if math.isnan(loss):
            break

        if savePath != None:
            print 'save model...'
            model.saveNetworks(savePath)

        dataset.newEpoch()
        epoch += 1

    veryEnd = time.time()
    sys.stdout.write("total training time:" + str(datetime.timedelta(seconds=veryEnd - veryStart)))

    # while epoch < training_epoch:
    #
    #     start = time.time()
    #     batchData = dataset.getNextBatch(batchSize=3)
    #     batchData['LearningRate'] = learningRate
    #     epochCurr = dataset._epoch
    #     dataStart = dataset._dataStart
    #     dataLength = dataset._dataLength
    #
    #
    #     if epochCurr != epoch or (int(iteration+1) % max_iter == 0 and (iteration + 1) != 1):
    #         print ''
    #         iteration = 0
    #         loss = loss * 0.0
    #         run_time = 0.0
    #         if savePath != None:
    #             print 'save model...'
    #             model.saveNetworks(savePath)
    #
    #     epoch = epochCurr
    #
    #     lossTemp = model.fit(batchData)
    #     end = time.time()
    #
    #     loss = float(loss * iteration + lossTemp) / float(iteration + 1.0)
    #     run_time = (run_time * iteration + (end - start)) / float(iteration + 1.0)
    #
    #     sys.stdout.write("Epoch:{:03d} iter:{:04d} runtime:{:.2f} ".format(int(epoch + 1), int(iteration + 1), run_time))
    #     sys.stdout.write("cur/tot:{:07d}/{:07d} ".format(dataStart, dataLength))
    #     sys.stdout.write("loss={:f} ".format(loss))
    #     sys.stdout.flush()
    #
    #     iteration = iteration + 1.0

# if __name__=="__main__":
#     sys.exit(trainMFNetDetector(
#         MFNetConfig=MFNetConfig,
#         batchSize=8,
#         training_epoch=10,
#         learningRate=0.0001,
#         savePath='weights/mfnet_detector',
#         restorePath='weights/mfnet_detector'
#     ))


if __name__=="__main__":
    sys.exit(trainMFNetDetector(
        MFNetConfig=MFNetConfig
        ,batchSize=32
        ,training_epoch=1
        ,max_iter=36000
        ,learningRate=1e-5
        ,savePath='weights/180729/4.lr1e-5iter36000'
        ,restorePath='weights/180729/3.lr1e-5iter10000'
        ))
        
