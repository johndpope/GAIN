# GAIN-tensorflow [under construction]

**Note: the code is not stable and some functions are still under construction**

The repository holds the implementation of [GAIN](https://arxiv.org/pdf/1802.10171.pdf), and the segmentation network mainly borrows the code from from [xtudbxk/SEC-tensorflow](https://github.com/xtudbxk/SEC-tensorflow).

## Prerequisite
 * Tensorflow >= 1.5
 * Download the Pascal VOC data and the pre-train VGG16 model, please refer to [step 2 in xtudbxk/SEC-tensorflow](https://github.com/xtudbxk/SEC-tensorflow#2-download-the-data-and-model)

## Train GAIN network
 * Training: `python [model].py -g <gpu_id> -f <gpu_fraction>`
 * Predicting Mask: `python [model].py -g <gpu_id> -f <gpu_fraction> -a inference`