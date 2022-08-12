import cv2 as cv
import pickle
from argparse import ArgumentParser
from data_utils import keypoints_mapping
from derive_hand import deriv_hands
import numpy as np

def draw_coords(frame, coords, point_i):
  
  font                   = cv.FONT_HERSHEY_SIMPLEX
  fontScale              = 1
  fontColor              = (255,255,255)
  thickness              = 2
  lineType               = 2
  
  cv.circle(frame,(int(coords[0]),int(coords[1])), 10, (70,70,200), -1)
  cv.putText(frame,'{}'.format(keypoints_mapping[point_i]), 
    (int(coords[0])+10,int(coords[1])+10), 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)

def draw_arrow(frame, coords, arrow, color):
  
  # to make smaller arrows
  arrow_modifier = 0.3
  arrow = arrow*arrow_modifier

  cv.arrowedLine(frame, (int(coords[0]),int(coords[1])), (int(coords[0]+arrow[0]),int(coords[1]+arrow[1])), color)


def draw_hand(frame, hand, speed=None, accel=None):

  show_speed = speed != None
  show_accel = accel != None

  for i, keypoint in enumerate(hand):
    draw_coords(frame, keypoint[0:2], i)
    if show_speed:
      draw_arrow(frame, keypoint[0:2], speed[i][0:2], (20,200,80))
    if show_accel:
      draw_arrow(frame, keypoint[0:2], accel[i][0:2], (200,80,20))

def draw_frame(frame, boxes, speeds, accels):
  
  for side, hand in boxes.items():

    passed_args = dict()

    if side in list(speeds.keys()):
      passed_args['speed'] = speeds[side]

    if side in list(accels.keys()):
      passed_args['accel'] = accels[side]

    draw_hand(frame, hand, **passed_args)
              
def main():

  parser = ArgumentParser()
  parser.add_argument(
    'videopath',
    help='video to draw on'
  )
  parser.add_argument(
    'coordspath',
    help='hand coordinates file path'
  )
  parser.add_argument(
    '-out-video',
    help='name of the outputed video',
    default='no_vid'
  )

  args = parser.parse_args()

  save_vid = True
  if args.out_video == 'no_vid':
    save_vid = False

  with open(args.coordspath,'rb') as f:
    coords_hands_dict = pickle.load(f)
    coords_hands = list(map(lambda x: {k: v['keypoints'] for k,v in x.items()}, coords_hands_dict))

  speeds = deriv_hands(coords_hands, 60)
  accels = deriv_hands(speeds, 1)

  i=-1
  cap = cv.VideoCapture(args.videopath)

  if save_vid:
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(args.out_video,fourcc,1.0,(1920,1080))

  while(cap.isOpened() and i+1<len(coords_hands)):
    ret, frame = cap.read()
    if ret == True:
      i+=1

      draw_frame(frame, coords_hands[i], speeds[i], accels[i])
      
      cv.imshow('Frame',frame)

      if save_vid:
        out.write(frame)

      if cv.waitKey(0) & 0xFF == ord('q'):
        break
  cap.release()

  if save_vid:
    out.release()
  cv.destroyAllWindows()



if __name__ == '__main__':
  main()