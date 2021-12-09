# Vision Transformer

_Authors: Wooseok Gwak_

This is the repository that implements Vision Transformer (Alexey Dosovitskiy et al, 2021) using tensorflow 2. The paper can be found at [here](https://arxiv.org/abs/2010.11929). 

The official Jax repository is [here](https://github.com/google-research/vision_transformer)


<img src="./images/vit.gif" width="500px"></img>


## Requirements

I recommend using python3 and conda virtual environment.

```bash
conda create -n myenv python=3.7
conda activate myenv
conda install --yes --file requirements.txt
```

After making a virtual environment, download the git repository and use the model for your own project. When you're done working on the project, deactivate the virtual environment with `conda deactivate`.

## Usage

```python
import tensorflow as tf
from model.model import model

vit = ViT(
    d_model = 50
    mlp_dim = 100,
    num_heads = 10,
    dropout_rate = 0.1,
    num_layers = 3,
    patch_size = 32,
    num_classes = 102
)

img = np.randn(1, 3, 256, 256)

preds = vit(img)
```

Because of dependeny problem for anaconda packages, I use tensorflow 2.3 and write the code for multi head attention. (the code can be found from [here](https://www.tensorflow.org/text/tutorials/transformer#multi-head_attention)) I recommend to use tf.keras.layers.MultiHeadAttention from tensorflow 2.5~.

## Training

```bash
python train.py
```

train.py is sample training code to verify whether it performs the desired operation. You can change the file to train the model on specific dataset.

- 2021.11.30 : WARNING:tensorflow:'gradients do not exist for variables' is not resolved!