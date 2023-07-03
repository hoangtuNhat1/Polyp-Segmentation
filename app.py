from logging import NullHandler
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import shutil
from PIL import Image
from lib.HarDMSEG import HarDMSEG
from utils.dataloader import test_dataset
import imageio
import stat

# Change the permissions of the directories to allow deletion
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='/content/drive/MyDrive/HarDNet-MSEG/snapshots/HarD-MSEG-best/HarD-MSEG-best.pth')

# Load the HarDMSEG model
opt = parser.parse_args()
model = HarDMSEG()
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

# Streamlit app
st.title("Lavie")
logo = Image.open("/content/drive/MyDrive/HarDNet-MSEG/logo-sun@2x.png")  # Replace "logo.png" with the path to your logo file
st.image(logo, use_column_width=True)

# File upload and submit button
file = st.file_uploader("Chọn một file ảnh", type=["png", "jpg", "jpeg"])
submit_button = st.button("Submit")
# Display the result
if file and submit_button:
  image= Image.open(file) 
  db_path = '/content/drive/MyDrive/HarDNet-MSEG/DB/{}/'.format(file.name)
  if os.path.exists(db_path):
    # If the folder exists, remove it and create a new one
    shutil.rmtree(db_path[:-1])
    os.makedirs(db_path)
  else:
    # If the folder does not exist, create a new one
    os.makedirs(db_path)
  imageio.imwrite(db_path + file.name, image)
  image = db_path + file.name
  save_path = '/content/drive/MyDrive/HarDNet-MSEG/Out/{}/'.format(file.name)
  opt = parser.parse_args()
  model = HarDMSEG()
  model.load_state_dict(torch.load(opt.pth_path))
  model.cuda()
  model.eval()
  if os.path.exists(save_path):
    # If the folder exists, remove it and create a new one
    shutil.rmtree(save_path[:-1])
    os.makedirs(save_path)
  else:
    # If the folder does not exist, create a new one
    os.makedirs(save_path)
  image_root = db_path
  gt_root = db_path
  test_loader = test_dataset(image_root, gt_root, opt.testsize)
  image, gt, name = test_loader.load_data()
  gt = np.asarray(gt, np.float32)
  gt /= (gt.max() + 1e-8)
  image = image.cuda()
  res = model(image)
  res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
  res = res.sigmoid().data.cpu().numpy().squeeze()
  res = (res - res.min()) / (res.max() - res.min() + 1e-8)
  res_uint8 = (res * 255).astype(np.uint8)
  imageio.imwrite(save_path + file.name, res_uint8)
  st.image(Image.open(save_path + file.name), caption="Kết quả Polyp Segmentation", use_column_width=True)
