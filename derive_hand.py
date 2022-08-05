import numpy as np
from data_utils import keypoints_mapping

def deriv_hands(hands_coords, framerate):

  hands = ['left', 'right']

  derived_hands = list()

  last_frame_present = dict()
  last_frame_present['left'] = False
  last_frame_present['right'] = False

  last_frame_coords = dict()
  last_frame_coords['left'] = None
  last_frame_coords['right'] = None

  for coords in hands_coords:

    frame_derived = dict()

    for hand in hands:

      keypoints_deriv = list()

      if last_frame_present[hand] and hand in list(coords.keys()):
        for i in range(len(keypoints_mapping)):
          keypoints_deriv.append((coords[hand][i]-last_frame_coords[hand][i])*framerate)
        frame_derived[hand] = keypoints_deriv

      last_frame_present[hand] = hand in list(coords.keys())
      if last_frame_present[hand]:
        last_frame_coords[hand] = coords[hand]

    derived_hands.append(frame_derived)

  return derived_hands