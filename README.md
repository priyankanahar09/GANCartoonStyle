# CartoonGAN

Here's a `README.md` file for your repository with the necessary steps to run your code in Google Colab:

```markdown
# CartoonGan-Tensorflow

This repository contains code for training and using a CartoonGAN model to transform images into cartoon-style images. The following instructions will guide you through setting up and running the model in Google Colab.

## Getting Started

To get started, open this notebook in Google Colab by clicking the link below:

[Open in Google Colab](https://colab.research.google.com/drive/1BhJpQMGSWOs3OQdjS5CcX5Yp77rEHz1M#scrollTo=KVK0kiqcUEh8)

### Clone the Repository

First, clone the repository and navigate to the project directory:

```python
import os
repo = "CartoonGan-tensorflow"

!git clone https://github.com/mnicnc404/{repo}.git
os.chdir(os.path.join(repo))
```


### Install Requirements

Install the necessary packages:

```python
!pip install -r requirements_gpu.txt
```

### Smooth Dataset Images

Run the script to smooth dataset images:

```python
!python '/content/CartoonGan-tensorflow/scripts/smooth.py' --path '/content/drive/MyDrive/Colab Notebooks/Datasets/GAN_DATASET'
```

### Train the Model

Start training the model with the specified parameters:

```python
import tensorflow as tf
tf.__version__

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
--dataset_name trainB
```

### Inference with Checkpoint

Run inference using a checkpoint:

```python
!python inference_with_ckpt.py \
  --light \
  --m_path '/content/CartoonGan-tensorflow/model' \
  --img_path '/content/CartoonGan-tensorflow/input_images/origami.jpg' \
  --out_dir '/content/CartoonGan-tensorflow/output_images/out'
```

### Export the Model

Export the trained model:

```python
!python export.py \
--m_path path/to/model/folder \
--out_dir /content/CartoonGan-tensorflow/exported_models  \
--light
```

### Inference with Saved Model

Run inference using the saved model:

```python
!python inference_with_saved_model.py \
    --m_path /content/CartoonGan-tensorflow/exported_models/light_paprika_SavedModel \
    --img_path /content/CartoonGan-tensorflow/input_images/origami.jpg \
    --out_dir /content/CartoonGan-tensorflow/output_images
```

### Additional Installations

Install additional packages:

```python
!pip install git+https://www.github.com/keras-team/keras-contrib.git
!pip install mcts
```

Convert the model to protobuf format:

```python
!python to_pb.py \
    --m_path  '/content/CartoonGan-tensorflow/exported_models' \
    --out_dir '/content/CartoonGan-tensorflow/export_model' \
    --light
```

### Testing

Clone the keras-contrib repository and set it up:

```python
!git clone https://www.github.com/keras-team/keras-contrib.git
!cd keras-contrib
!python convert_to_tf_keras.py
!USE_TF_KERAS=1 python setup.py install
```

### Cartoonize an Image

Download an image and apply the cartoon style:

```python
from IPython.display import clear_output, display, Image
import time
import os

# URL of the image to be cartoonized
image_url = 'https://media.giphy.com/media/o5HKScC1PflLO/giphy.gif'

input_image_dir = "input_images"
output_image_dir = input_image_dir.replace("input_", "output_")

if image_url:
    img_filename = image_url.split("/")[-1]
    name, ext = '.'.join(img_filename.split(".")[:-1]), img_filename.split(".")[-1]
    new_name = '_'.join((name, str(int(time.time()))))
    new_img_filename = '.'.join((new_name, ext))
    image_path = os.path.join(input_image_dir, new_img_filename)

    !wget {image_url}
    !mv {img_filename} {new_img_filename}
    !mv {new_img_filename} {image_path}

# Convert gif to png if necessary
if ".gif" in new_img_filename:
    png_path = new_img_filename + '.png'
    !cp {image_path} {png_path}

display(Image(png_path))

styles = "hayao"  # Select style: shinkai, hayao, hosoda, paprika

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

include_original_image = "no"  # Change to "yes" to include original image in the output

from google.colab import files
if include_original_image == "yes":
    output_image_path = os.path.join(output_image_dir, "comparison", new_img_filename)
else:
    output_image_path = os.path.join(output_image_dir, styles, new_img_filename)

files.download(output_image_path)
```

