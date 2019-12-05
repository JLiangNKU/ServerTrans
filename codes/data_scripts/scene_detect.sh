#!/bin/bash

SRC_DIR=/home/xiyang/Datasets/4KHDR/SDR_4K
DST_DIR=/home/xiyang/Datasets/4KHDR/video_scenes_thres35
mkdir $DST_DIR
FILES=$(ls $SRC_DIR | grep .mp4)

for FILE in $FILES
do
    FILENAME="${FILE:0:-4}"
    echo $FILENAME
    scenedetect --input $SRC_DIR/$FILE --output $DST_DIR list-scenes detect-content -t 35
done