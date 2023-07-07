# Polyp Segmentation 
# **Any question? Contact me via Facebook link on Profile Page. I'm willing to share any time**

A Streamlit Web application to demonstrate Polyp Segmentation using HarDNet-MSEG

Here the web demo video: 

https://drive.google.com/file/d/1KOywBzb8HY3CHHSbp85b5yv62MPfahai/view?usp=sharing

Run web immediately from your computer:

First, Download or clone the project to your machine

![image](https://github.com/PhamQuangNhut/HarDNet-MSEG/assets/88762631/40c0f089-a891-4062-83f8-8a1c97221491)

Then, Put it to folder name HarDNet-MSEG and upload the folder on your Google Drive for Drive Mounting Later

![image](https://github.com/PhamQuangNhut/HarDNet-MSEG/assets/88762631/45177295-4b3d-4b29-96c3-24b390a5f32e)

Download .pth file from [https://drive.google.com/file/d/12KoEyko7-dBnAxi_IDMWpHSLPYDHhe6s/view?usp=drive_link] and add it to your drive folder, then modified the --pth_path in line 18 app.py file

Open and run file app.ipynb by Google Colab

![image](https://github.com/PhamQuangNhut/HarDNet-MSEG/assets/88762631/434b6b52-4d2d-4ce3-9b28-d5f14abf4d4e)

Open Streamlit web and use it !

If use want to experience from the model building step, Here the tutorial: 

# 1. Training/Testing
  - Environment setting (Prerequisites):
    
    + Google Colab comes pre-integrated with PyTorch for you.
      
  - Downloading necessary data:
    + The project already have all the dataset you need but you can check for the source and download it for here
    **For Kvasir-SEG Dataset reference from**
    [**Real-Time Polyp Detection, Localisation and Segmentation in Colonoscopy Using Deep Learning**](https://arxiv.org/abs/2011.07631)
    **(Only training using Kvasir-SEG, 880 images for training 120 images for testing)**
    
     + downloading testing dataset and move it into your test_path
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1us5iOMWVh_4LAiACM-LQa73t1pLLPJ7l/view?usp=sharing).
    
    + downloading training dataset and move it into your train_path
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/17sUo2dLcwgPdO_fD4ySiS_4BVzc3wvwA/view?usp=sharing).
    **For each Dataset training including Kvasir-SEG, CVC-ColonDB, EndoScene, ETIS-Larib Polyp DB and CVC-Clinic DB from**
    [**PraNet: Parallel Reverse Attention Network for Polyp Segmentation**](https://arxiv.org/abs/2006.11392)
    
    + downloading testing dataset and move it into your test_path
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1o8OfBvYE6K-EpDyvzsmMPndnUMwb540R/view?usp=sharing).
    
    + downloading training dataset and move it into your train_path
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1lODorfB33jbd-im-qrtUgWnZXxB94F55/view?usp=sharing).
- Training :
  
    1. First download pretrain_weight : hardnet68.pth for HarDNet68 in https://github.com/PingoLH/Pytorch-HarDNet  (Already in the project)
    
    2. Change the weight path in lib/hardnet_68.py line 208 for loading the pretrain_weight  
    
    3. Change the --train_path & --test_path in Train.py  
    
    4. Final step is to run the Train.py  

- Testing & inference result :

    1. Change the data_path in Test.py (line 16) 
    
    2. Here is the weight we trained for Kvasir-SEG using on the report (https://drive.google.com/file/d/12KoEyko7-dBnAxi_IDMWpHSLPYDHhe6s/view?usp=drive_link)   
    
       Download it, and run "python Test.py --pth_path "path of the weight"    
    
       And you can get the inference results in results/
    

### Evaluation :

1. Change the image_root, gt_root in line 49, 50 in eval_Kvasir.py  
2. Run the eval_Kvasir.py to get a similar result (about +0.002) to our report for Kvasir Dataset.  


