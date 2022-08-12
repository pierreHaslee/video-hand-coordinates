import numpy as np

keypoints_mapping = [
  'wrist',
  'thumb4',
  'thumb3',
  'thumb2',
  'thumb1',
  'index4',
  'index3',
  'index2',
  'index1',
  'middle4',
  'middle3',
  'middle2',
  'middle41',
  'ring4',
  'ring3',
  'ring2',
  'ring1',
  'little4',
  'little3',
  'little2',
  'little1',
]

def combine_results(boxes, poses):
  for i in range(len(boxes)):
    poses[i]['bbox'] = boxes[i]['bbox']
  return poses

def coords_to_hands(coords_list, frame_x_length):

  coords_hands_video = list()

  for coords_frame in coords_list:

    #work for each frame
    coords_hands = get_coords_hands(coords_frame, frame_x_length)
    coords_hands_video.append(coords_hands)

  return coords_hands_video


def get_coords_hands(coords_frame, frame_x_width):

  coords_hands = dict()

  barycentres = list(map(lambda x : hand_barycentre(x['keypoints'], only_x=True), coords_frame))
  for bar in barycentres:
    print(bar)

  if len(barycentres) > 0:
    if len(barycentres) == 1:
      if barycentres[0] < frame_x_width/2:
        coords_hands['left'] = coords_frame[0]
      else:
        coords_hands['right'] = coords_frame[0]
    else:
      id_left = barycentres.index(min(barycentres))
      coords_hands['left'] = coords_frame[id_left]
      id_right = barycentres.index(max(barycentres))
      coords_hands['right'] = coords_frame[id_right]
  
  return coords_hands


def hand_barycentre(hand_keypoints, only_x=False):

  keyp = np.array(hand_keypoints)
  barycentre = keyp.mean(axis=0)[0:1]

  if only_x:
    return barycentre[0]

  return barycentre