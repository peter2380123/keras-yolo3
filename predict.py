#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np

import csv # to write csv output

#def _main_(args):
def _main_(filename):
    config_path  = "config.json"
    #input_path   = args.input
    input_path = filename
    output_path  = "result/"

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.7, 0.25 #default is 0.5 and 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes 
    ###############################
    if 'webcam' in input_path: # do detection on the first webcam
        video_reader = cv2.VideoCapture(0)

        # the main loop
        batch_size  = 1
        images      = []
        curr_frame = 0
        while True:
            ret_val, image = video_reader.read()
            if ret_val == True: images += [image]
            curr_frame += 1

            if (len(images)==batch_size) or (ret_val==False and len(images)>0):
                batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                for i in range(len(images)):
                    draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh, True, curr_frame) 
                    cv2.imshow('video with bboxes', images[i])
                images = []
            if cv2.waitKey(1) == 27: 
                break  # esc to quit
        cv2.destroyAllWindows()        
    elif input_path[-4:] == '.mp4' or input_path[-4:] == '.MP4': # do detection on a video  
        video_out = output_path + input_path.split('/')[-1]
        video_reader = cv2.VideoCapture(input_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))
        # the main loop
        batch_size  = 1
        images      = []
        start_point = 0 #%
        show_window = False
        f_name_extract = filename.split('/')
        csv_path = output_path + f_name_extract[len(f_name_extract)-1] + '.csv'
        with open(csv_path, mode='w') as csv_file:
            fields = ['FrameNumber', 'PredictionString (class,conf,xmin,ymin,xmax,ymax)']
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()

            for i in tqdm(range(nb_frames)):
                curr_frame = i+1 # so first frame is named frame 1 not frame 0
                _, image = video_reader.read()

                if (float(i+1)/nb_frames) > start_point/100.:
                    images += [image]

                    if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
                        # predict the bounding boxes
                        batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                        for i in range(len(images)):
                            # draw bounding boxes on the image using labels
                            _, info = draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh, True, curr_frame)   
                            splitted = info.split(' ', 1) # split only on the first occurrence of space
                            writer.writerow({'FrameNumber':splitted[0], 'PredictionString (class,conf,xmin,ymin,xmax,ymax)':splitted[1]})

                            # show the video with detection bounding boxes          
                            if show_window: cv2.imshow('video with bboxes', images[i])  

                            # write result to the output video
                            video_writer.write(images[i]) 
                        images = []
                    if show_window and cv2.waitKey(1) == 27: break  # esc to quit

            if show_window: cv2.destroyAllWindows()
            video_reader.release()
            video_writer.release()      
        # end open csv 
    else: # do detection on an image or a set of images
        image_paths = []

        if os.path.isdir(input_path): 
            for inp_file in os.listdir(input_path):
                image_paths += [input_path + inp_file]
        else:
            image_paths += [input_path]

        image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]

        # the main loop
        for image_path in image_paths:
            image = cv2.imread(image_path)
            print(image_path)

            # predict the bounding boxes
            boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)[0]

            # draw bounding boxes on the image using labels
            _, info = draw_boxes(image, boxes, config['model']['labels'], obj_thresh) 

            # write the image with bounding boxes to file
            cv2.imwrite(output_path + image_path.split('/')[-1], np.uint8(image))         

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    
    args = argparser.parse_args()
    _main_(args)
