import torch
from torch.autograd import Variable
import torch.functional as F
import dataLoader
import argparse
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import model
import torch.nn as nn
import os
import numpy as np
import utils
import scipy.io as io

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--imageRoot', default='../data/VOC2012_train_val/VOC2012_train_val/JPEGImages', help='path to input images' )
parser.add_argument('--labelRoot', default='../data/VOC2012_train_val/VOC2012_train_val/SegmentationClass', help='path to input images' )
parser.add_argument('--fileList', default='../data/VOC2012_train_val/VOC2012_train_val/ImageSets/Segmentation/train.txt', help='path to input images' )
parser.add_argument('--experiment', default='train', help='the path to store sampled images and models' )
parser.add_argument('--epochId', type=int, default=210, help='the number of epochs being trained')
parser.add_argument('--batchSize', type=int, default=32, help='the size of a batch' )
parser.add_argument('--numClasses', type=int, default=21, help='the number of classes' )
parser.add_argument('--isDilation', action='store_true', help='whether to use dialated model or not' )
parser.add_argument('--initLR', type=float, default=0.1, help='the initial learning rate')
parser.add_argument('--iterationDecreaseLR', type=int, nargs='+', default=[1500000, 2500000], help='the iteration to decrease learning rate')
parser.add_argument('--isSpp', action='store_true', help='whether to do spatial pyramid or not' )
parser.add_argument('--ASpp', action='store_true', help='whether to do ASPP or not' )
parser.add_argument('--noCuda', action='store_true', help='do not use cuda for training' )
parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network' )
parser.add_argument('--colormap', default='colormap.mat', help='colormap for visualization')



if __name__=="__main__":
    # The detail network setting
    opt = parser.parse_args()
    print(opt)

    colormap = io.loadmat(opt.colormap )['cmap']


    if opt.isSpp == True :
        opt.isDilation = False

    if opt.isDilation:
        opt.experiment += '_dilation'

    if opt.isSpp:
        opt.experiment += '_spp'

    if opt.ASpp:
        opt.experiment += '_aspp'
        opt.isDilation = False
        opt.isSpp == False


    # Save all the codes
    os.system('mkdir %s' % opt.experiment )
    os.system('cp *.py %s' % opt.experiment )

    if torch.cuda.is_available() and opt.noCuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # Initialize image batch
    imBatch = Variable(torch.FloatTensor(opt.batchSize, 3, 300, 300) )
    labelBatch = Variable(torch.FloatTensor(opt.batchSize, opt.numClasses, 300, 300) )
    maskBatch = Variable(torch.FloatTensor(opt.batchSize, 1, 300, 300) )
    labelIndexBatch = Variable(torch.LongTensor(opt.batchSize, 1, 300, 300) )

    # Initialize network
    if opt.isDilation:
        encoder = model.encoderDilation()
        decoder = model.decoderDilation()
    elif opt.isSpp:
        encoder = model.encoderSPP()
        decoder = model.decoderSPP()
    elif opt.ASpp:
        encoder = model.encoderASPP()
        decoder = model.decoderASPP()
    else:
        encoder = model.encoder()
        decoder = model.decoder()

    model.loadPretrainedWeight(encoder)
    model.loadPretrainedWeight(decoder)
    # Move network and containers to gpu
    if not opt.noCuda:
        device = 'cuda'
    else:
        device = 'cpu'
        
        
    # Initialize optimizer
    optimizer = optim.SGD([
                    {'params': encoder.parameters(), 'lr': opt.initLR/5},
                    {'params': decoder.parameters()}
                ], lr=opt.initLR, momentum=0.9)

    imBatch = imBatch.to(device)
    labelBatch = labelBatch.to(device)
    labelIndexBatch = labelIndexBatch.to(device)
    maskBatch = maskBatch.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Initialize dataLoader
    segDataset = dataLoader.BatchLoader(
            imageRoot = opt.imageRoot,
            labelRoot = opt.labelRoot,
            fileList = opt.fileList,
            imWidth = 224, imHeight = 224
            )
    segLoader = DataLoader(segDataset, batch_size=opt.batchSize, num_workers=1, shuffle=True )

    lossArr = []

    iteration = 0
    epoch = opt.epochId
    confcounts = np.zeros( (opt.numClasses, opt.numClasses), dtype=np.int64 )
    accuracy = np.zeros(opt.numClasses, dtype=np.float32 )

    for epoch in range(0, opt.epochId ):
        trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
        for i, dataBatch in enumerate(segLoader ):
            iteration += 1

            # Read data
            imBatch = Variable(dataBatch['im']).to(device)
            labelBatch = Variable(dataBatch['label']).to(device)
            labelIndexBatch = Variable(dataBatch['labelIndex']).to(device)
            maskBatch = Variable(dataBatch['mask']).to(device)

            # Test network
            optimizer.zero_grad()
            x1, x2, x3, x4, x5 = encoder(imBatch )
            pred = decoder(imBatch, x1, x2, x3, x4, x5 )

            # Compute mean IOU
            loss = torch.mean( pred * labelBatch )
            loss.backward()
            optimizer.step()

            hist = utils.computeAccuracy(pred, labelIndexBatch, maskBatch )
            confcounts += hist

            for n in range(0, opt.numClasses ):
                rowSum = np.sum(confcounts[n, :] )
                colSum = np.sum(confcounts[:, n] )
                interSum = confcounts[n, n]
                accuracy[n] = float(100.0 * interSum) / max(float(rowSum + colSum - interSum ), 1e-5)

            # Output the log information
            lossArr.append(loss.cpu().data.item() )
            meanLoss = np.mean(np.array(lossArr[:] ) )
            meanAccuracy = np.mean(accuracy )

            print('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f'  \
                    % ( epoch, iteration, lossArr[-1], meanLoss ) )
            print('Epoch %d iteration %d: Accumulated Accuracy %.5f' \
                    % ( epoch, iteration, meanAccuracy ) )
            trainingLog.write('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f \n' \
                    % ( epoch, iteration, lossArr[-1], meanLoss ) )
            trainingLog.write('Epoch %d iteration %d: Accumulated Accuracy %.5f \n' \
                    % ( epoch, iteration, meanAccuracy ) )
            
            if iteration in opt.iterationDecreaseLR:
                print('The learning rate is being decreased at iteration %d' % iteration )
                trainingLog.write('The learning rate is being decreased at iteration %d\n' % iteration )
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 10
        
        trainingLog.close()
    # Save the accuracy
        if (epoch+1) % 5 == 0:
            np.save('%s/loss.npy' % (opt.experiment), np.array(lossArr[:] ) )
            np.save('%s/accuracy.npy' % (opt.experiment), accuracy )
            torch.save(encoder.state_dict(), '%s/encoder_%d.pth' % (opt.experiment, epoch+1) )
            torch.save(decoder.state_dict(), '%s/decoder_%d.pth' % (opt.experiment, epoch+1) )
        

    