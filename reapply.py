import numpy as np
import os
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
from utils.colors import get_color

import csv


def _main_(video_path, csv_path):
  # config_path  = "config.json"
  input_path   = video_path
  output_path  = "result/reapply/"

  makedirs(output_path)

  ###############################
  #   Predict bounding boxes 
  ###############################
  if input_path[-4:] == '.mp4' or input_path[-4:] == '.MP4': # do detection on a video  
    with open(csv_path, 'r') as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      rows = list(csv_reader)

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
    show_window = False #RESET back to false
    for i in tqdm(range(nb_frames)):
      curr_frame = i+1
      _, image = video_reader.read()
      if (float(i+1)/nb_frames) > start_point/100.:
        images += [image]

        if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
          # predict the bounding boxes
          #batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

          for i in range(len(images)):
            # draw bounding boxes on the image using labels
            #draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)   

            imgh, imgw, channels = images[i].shape

            box_info = rows[curr_frame][1].split()
            label, conf = box_info[0], box_info[1]

            lab_colors = ["dolphin", "human", "shark"]
            obj_color = lab_colors.index(label)
            
            xmin = round(float(box_info[2])*imgw)
            xmax = round(float(box_info[4])*imgw)
            ymin = round(float(box_info[3])*imgh)
            ymax = round(float(box_info[5])*imgh)

            #xmin, ymin = round(float(box_info[2])*imgw), round(float(box_info[3])*imgh)
            #xmax, ymax = round(float(box_info[4])*imgw), round(float(box_info[5])*imgh)

            label_str = label + ' ' + conf

            text_size = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 1.1e-3 * image.shape[0], 5)
            width, height = text_size[0][0], text_size[0][1]

            region = np.array([[xmin-3,        ymin], 
                               [xmin-3,        ymin-height-26], 
                               [xmin+width+13, ymin-height-26], 
                               [xmin+width+13, ymin]], dtype='int32')

            cv2.rectangle(img=image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=get_color(obj_color), thickness=3)
            cv2.fillPoly(img=image, pts=[region], color=get_color(obj_color))
            cv2.putText(img=image, 
                        text=label_str, 
                        org=(xmin+13, ymin - 13), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1e-3 * image.shape[0], 
                        color=(0,0,0), 
                        thickness=2) 

            # show the video with detection bounding boxes          
            if show_window: cv2.imshow('video with bboxes', images[i])  

            # write result to the output video
            video_writer.write(images[i]) 
          images = []
        if show_window and cv2.waitKey(1) == 27: break  # esc to quit

    if show_window: cv2.destroyAllWindows()
    video_reader.release()
    video_writer.release()
  else: 
    return # not a video
