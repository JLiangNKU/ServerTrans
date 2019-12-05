#!/bin/bash

SRC_DIR=/data1/yangxi/AI_4K/SDR_4K
DST_DIR=/data1/yangxi/AI_4K/SDR_4K_PNG

FILES=$(ls $SRC_DIR | grep .mp4)

for FILE in $FILES
do
    FILENAME="${FILE:0:-4}"
    echo $FILENAME
    mkdir $DST_DIR/$FILENAME
    ffmpeg -i $SRC_DIR/$FILE $DST_DIR/$FILENAME/%03d.png
done
