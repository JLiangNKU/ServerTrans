#!/bin/bash

SRC_DIR=540p_test
DST_DIR=RCANplus_video

FILES=$(ls $SRC_DIR)

for FILE in $FILES
do
    echo $FILE
    #mkdir $DST_DIR/$FILENAME
    ffmpeg -framerate 24 -i $SRC_DIR/$FILE/frame_%3d.png -vcodec libx265 -crf 6 -pix_fmt yuv422p $DST_DIR/$FILE.mp4
done
