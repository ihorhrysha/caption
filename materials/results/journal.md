<!--- ###################################################### --->

# [e0005] Experiment with decoder model

<!--- ###################################################### --->

# [e0003] Use flickr dataset

<!--- ###################################################### --->

# [e0002] Change encoder model architecture

<!--- ###################################################### --->

# [e0001] Training baseline model

|Start Date|End Date  |
|----------|----------|
|2021-12-30|2021-12-30|

## Motivation

## Description

## Deliverables
  
## Interpretation

<!--- ###################################################### --->

# [e0000] Image captioning experimentation framework

|Start Date|End Date  |
|----------|----------|
|2021-12-30|2021-12-30|

## Motivation
Use given framework on a specific problem

## Description
In most cases for image captioning encoder-decoder approach is used. The encoder is some pretrained CNN architecture and decoder is some recurrent NN. I found a [repo](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning) of a model which uses ResNet-152 as encoder and LSTM layer(s) as decoder. It looks pretty straightforward and is nice candidate for the baseline model.

It took a long time to get used to the experiment framework, but I must say that the structured configurable approach with nice logging makes sense. Most of the time I spent adopting dataset classes for a specific task. I downloaded coco 2014 dataset(as this is the only available edition with image captioning annotations). I started using coco py module, then I found coco in torchvision.datasets but it was not quite useful as it does not store the vocabulary. So I decided to use custom dataset implementation.


Fixing model and configuration modules were quite easy. Most of the utility functions were used without any changes. After a few attempts I managed to kick off the training task. But there are still two main issues: 
1. There is no small dataset for image caption training - so powerful hardware with GPU is needed.
2. I haven't implemented any metric(s). There are some like Rouge and Bleu but they mainly use intersection between words in two sentences. So some manual evaluation will be needed after each experiment

## Deliverables
- Experimentation framework refactored and extended for image captioning problem
- Tested on pretrained model
- Implemented data class for COCO image caption dataset

## TODOS
- Define metric(s)
- Check model and experiment analysis notebooks