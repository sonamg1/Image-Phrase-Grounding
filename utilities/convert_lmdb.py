'''
Code to generate lmdb database from image folder

'''


import numpy as np
import lmdb
import os
import cv2
import argparse

parser = argparse.ArgumentParser(description='Converts a list of image files to an lmdb databse.')
parser.add_argument('--img_dir', type=str, required=True, help='Directory where unzipped image files are.')
parser.add_argument('--lmdb_filename', type=str, required=True, help='Path where the lmdb database needs to be created.')
parser.add_argument('--max_files', type=int, default=None, help='Number of files to convert. If not set, converts all.')
args = parser.parse_args()

env = lmdb.open(args.lmdb_filename,  map_size=int(1e11))

max_files = args.max_files
num_files = 0

with env.begin(write=True) as txn:
      for f in os.listdir(args.img_dir):
        file_name = os.path.join(args.img_dir, f)
        if os.path.isfile(file_name):
          # Assuming the filename looks like COCO_<split_name>_<id>.jpg. We pick the last 6 digits from <id>.
          key = f.split('.')[0].split('_')[2][-6:]
          img = cv2.imread(file_name, cv2.IMREAD_COLOR)
          # PNG encoding is lossless but its 4 times larger. 
          ret, encoded_img = cv2.imencode('.jpg', img) 
          txn.put(key.encode('utf-8'), encoded_img)
          num_files += 1
          if num_files % 1000 == 0:
            print('Processed: ', num_files)
          if max_files is not None and num_files == max_files:
            break
      
print('Done')
