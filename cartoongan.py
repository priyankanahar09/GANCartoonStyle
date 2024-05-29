# -*- coding: utf-8 -*-
"""CartoonGAN.ipynb


import os #
repo = "CartoonGan-tensorflow"

!git clone https://github.com/mnicnc404/{repo}.git
os.chdir(os.path.join(repo))

from google.colab import drive
drive.mount('/content/drive')

!pip install -r requirements_gpu.txt

!python '/content/CartoonGan-tensorflow/scripts/smooth.py' --path '/content/drive/MyDrive/Colab Notebooks/Datasets/GAN_DATASET'
#In order to generate trainB_smooth, can run scripts/smooth.py

import tensorflow as tf
tf.__version__

# Commented out IPython magic to ensure Python compatibility.

# %cd /content/CartoonGan-tensorflow


!python train.py \
--light \
--batch_size 8 \
--pretrain_epochs 1 \
--content_lambda .4 \
--pretrain_learning_rate 2e-4 \
--g_adv_lambda 8. \
--generator_lr 8e-5 \
--discriminator_lr 3e-5 \
--style_lambda 25. \
--dataset_name {trainB}

!python inference_with_ckpt.py \
  --light \
  --m_path '/content/CartoonGan-tensorflow/model' \
  --img_path '/content/CartoonGan-tensorflow/input_images/origami.jpg' \
  --out_dir '/content/CartoonGan-tensorflow/output_images/out'

!python export.py \
--m_path path/to/model/folder \
--out_dir /content/CartoonGan-tensorflow/exported_models  \
--light

!python inference_with_saved_model.py \
    --m_path /content/CartoonGan-tensorflow/exported_models/light_paprika_SavedModel \
    --img_path /content/CartoonGan-tensorflow/input_images/origami.jpg \
    --out_dir /content/CartoonGan-tensorflow/output_images

!pip install git+https://www.github.com/keras-team/keras-contrib.git
!pip install mcts

!python to_pb.py \
    --m_path  '/content/CartoonGan-tensorflow/exported_models' \
    --out_dir '/content/CartoonGan-tensorflow/export_model' \
    --light

"""**Testing**"""

!ls | grep cartoonize.py

from IPython.display import clear_output, display, Image

#!pip install tensorflow-gpu==2.0.0-alpha0
!git clone https://www.github.com/keras-team/keras-contrib.git \
    !cd keras-contrib \
    !python convert_to_tf_keras.py \
    !USE_TF_KERAS=1 python setup.py install
clear_output()

# URL
image_url = 'https://media.giphy.com/media/o5HKScC1PflLO/giphy.gif'

input_image_dir = "input_images" #
output_image_dir = input_image_dir.replace("input_", "output_") #

import time
if image_url:
    img_filename = image_url.split("/")[-1] #
    name, ext = '.'.join(img_filename.split(".")[:-1]), img_filename.split(".")[-1] #
    new_name = '_'.join((name, str(int(time.time())))) #
    new_img_filename = '.'.join((new_name, ext)) #
    image_path = os.path.join(input_image_dir, new_img_filename) #


    !wget {image_url} \
        && mv {img_filename} {new_img_filename} \
        && mv {new_img_filename} {image_path}

#  gif в png
if ".gif" in new_img_filename:
    png_path = new_img_filename + '.png'
    !cp {image_path} {png_path}

display(Image(png_path)) # png

#
styles = "hayao" #@param ["shinkai", "hayao", "hosoda", "paprika"]

!python cartoonize.py \
  --styles {styles} \
  --comparison_view horizontal

if img_filename:
    if ".gif" in img_filename:
        generated_gif = os.path.join(output_image_dir, "comparison", new_img_filename)
        result_path = generated_gif + '.png'
        !cp {generated_gif} {result_path}
    else:
        result_path = os.path.join(output_image_dir, "comparison", new_img_filename)

display(Image(result_path))

include_original_image = "no"  #@param {type: "string"}

from google.colab import files
if include_original_image == "yes":
    output_image_path = os.path.join(output_image_dir, "comparison", new_img_filename)
else:
    output_image_path = os.path.join(output_image_dir, styles, new_img_filename)
files.download(output_image_path)