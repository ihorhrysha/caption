#!/bin/bash
# KAGGLE_USERNAME=xxx KAGGLE_KEY=yyy kaggle datasets download --unzip  -p ./flickr8k adityajn105/flickr8k

# after kaggle is installed locally run
DATASET_NAME=flickr8k

kaggle datasets download --unzip  -p ./$DATASET_NAME adityajn105/flickr8k

mkdir ${DATASET_NAME}/vocab