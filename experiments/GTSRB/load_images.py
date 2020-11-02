import os
import pandas as pd 
import numpy as np
import cv2

class load_images:
  def __init__(self, folder_path, csv_name):
    self.path = folder_path
    self.df = pd.read_csv(csv_name, sep=';')

  def load(self, adaptive=True, size=(32, 32)):
    images = []
    labels = []
    for idx, row in self.df.iterrows():
      images.append([self.preprocess(cv2.imread(os.path.join(self.path, row['Filename'])),
                                      row=row, adaptive=adaptive, size=size)])
      labels.append([int(row['ClassId'])])
    
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    return images, labels

  def preprocess(self, im, row, adaptive, size):
    im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)

    if adaptive:
      # コントラスト制限適応ヒストグラム平坦化 (CLAHE)
      clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
      im_yuv[:,:,0] = clahe.apply(im_yuv[:,:,0])
    else:
      im_yuv[:,:,0] = cv2.equalizeHist(im_yuv[:,:,0])

    im = cv2.cvtColor(im_yuv, cv2.COLOR_YUV2RGB)
    im = im[int(row['Roi.X1']):int(row['Roi.X2']), int(row['Roi.Y1']):int(row['Roi.Y2']), :]
    im = cv2.resize(im, size)
    return im
