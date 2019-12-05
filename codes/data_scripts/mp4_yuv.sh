#!/bin/bash

SRC_DIR=/home/xiyang/Datasets/4KHDR/SDR_540p
DST_DIR=/home/xiyang/Datasets/4KHDR/SDR_540p_YUV

FILES=$(ls $SRC_DIR | grep .mp4)

for FILE in $FILES
do
    FILENAME="${FILE:0:-4}"
    echo $FILENAME
    ffmpeg -i $SRC_DIR/$FILE $DST_DIR/$FILENAME.yuv
done