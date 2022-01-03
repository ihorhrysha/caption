#!/bin/bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd ${CURRENT_DIR}

FILE_TRAIN=train2014.zip
FILE_VAL=val2014.zip
FILE_CAPTION=annotations_trainval2014.zip

DATASET_NAME=coco

URL_TRAIN=http://images.cocodataset.org/zips/train2014.zip
URL_VAL=http://images.cocodataset.org/zips/val2014.zip
URL_CAPTION=http://images.cocodataset.org/annotations/annotations_trainval2014.zip

if ! [ -d ${DATASET_NAME} ]; then
        
    mkdir $DATASET_NAME

    echo "* downloading caption annotations ..."
    wget ${URL_CAPTION}
    unzip $FILE_CAPTION -d $DATASET_NAME
    rm $FILE_CAPTION    

    echo "* downloading val dataset ..."
    wget ${URL_VAL}
    unzip $FILE_VAL -d $DATASET_NAME
    rm $FILE_VAL

    echo "* downloading train dataset ..."
    wget ${URL_TRAIN}
    unzip $FILE_TRAIN -d $DATASET_NAME
    rm $FILE_TRAIN

else
    echo "file exists"
fi


