import pickle
from argparse import ArgumentParser

from derive_hand import deriv_hand

parser = ArgumentParser()
parser.add_argument(
  'open_path',
  help='the pickle file to parse'
)
parser.add_argument(
  '--derived',
  help='the file you want to open is the derivation of a hand',
  action= 'store_true',
  default=False
)
parser.add_argument(
  '-hand',
  help='the derived hand you want to read',
  type=str
)

args = parser.parse_args()

list_hands = list()

with open(args.open_path, 'rb') as f:
  list_hands = pickle.load(f)

# list_results format:
# list of frames
#   list of bboxes
#     dict containing bbox coordinates and keypoints coordinates

if args.derived:

  list_hands = deriv_hand(args.hand, list_hands, 60)

  for i,frame in enumerate(list_hands):
    print('in frame num {}'.format(i))
    for coords in frame:
      print('{} hand keypoints:\n{}'.format(args.hand,coords))

else:
  for i,frame in enumerate(list_hands):
    print('in frame num {}'.format(i))
    for hand,coords in frame.items():
      print('{} hand keypoints:\n{}'.format(hand,coords))