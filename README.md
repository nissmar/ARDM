# ARDM
MVA Generative Models project on [Order Agnostic Autoregressive Diffusion Models](https://github.com/google-research/google-research/tree/master/autoregressive_diffusion).

## Face generation ([CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html))

<img src="examples/a.jpeg" alt="img" width="100"/> <img src="examples/b.jpeg" alt="img" width="100"/> <img src="examples/c.jpeg" alt="img" width="100"/><img src="examples/d.jpeg" alt="img" width="100"/> <img src="examples/e.jpeg" alt="img" width="100"/> 


## Character Generation ([binary MNIST](https://github.com/aiddun/binary-mnist))

<img src="examples/1_gen.png" alt="img" width="100"/> <img src="examples/2_gen.png" alt="img" width="100"/> <img src="examples/3_gen.png" alt="img" width="100"/><img src="examples/4_gen.png" alt="img" width="100"/> <img src="examples/5_gen.png" alt="img" width="100"/> 

## Source code

- Run `MNIST.ipynb` to train and evaluate a UNet model on the binary MNIST dataset
- Run `TinyCelebA.ipynb` to train and evaluate a UNet model on the Tiny CelebA dataset (we use 60×73 images with 32 grey-levels)
- Run `TuringTest.ipynb` to compare generated images with images from the dataset

## Code reuse 

We used parts from [UNet](https://github.com/zhixuhao/unet), [oardm](https://github.com/DuaneNielsen/oardm) and [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
