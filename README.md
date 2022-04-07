# Unet-and-Variants
Deployed resnet18 block for encoder-decoder. Modified the encoder with dilation convolution, pyramid pooling layer
and atrous spatial pyramid pooling (ASPP) separately.  
After training on the PASCAL VOC 2012 dataset, the mean IOU of variant adding ASPP is 14.7% higher than the
basic, from 50.96 to 58.45
