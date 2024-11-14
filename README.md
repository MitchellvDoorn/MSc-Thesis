# MSc-Thesis
Code for my Master Thesis, where Physics Informed Neural Networks are applied to optimization of Spacecraft Trajectories, specifically gravity assists.

1. [Installation](#installation)
2. [Basic PCNN model manual](#run_basic_pcnn)

## 1. Installation <a name="installation"></a>

#### Create a virtual environment
Make sure anaconda or miniconda is installed on your device.
Then create a virtual anaconda environment with Python 3.10.

```
conda create -n "pcnn-env" python=3.10
```

#### Dependencies
Install [tudatpy](https://docs.tudat.space/en/latest/). This astrodynamics toolbox is used for verification of the spacecraft trajectories found by the PCNN.

```
conda install -c tudat-team tudatpy
```

Install [deepxde](https://deepxde.readthedocs.io/en/latest/user/installation.html), a neural network module speciliased in Physics-Informed Neural Networks.

```
pip install deepxde
```

Install Tensorflow and Tensorflow probability

```
pip install tensorflow
pip install tensorflow-probability
```

Clone my repository. 
In the anaconda prompt, navigate to the directory where you want the repository to be saved, then clone it.

```
git clone https://github.com/MitchellvDoorn/MSc-Thesis
```

Now you should be able to run all of my code.

To run the basic pcnn example I made in Jupyter Notebook, you obviously need jupyter notebook. Therefore install it:

```
pip install notebook
```


## 2. Basic PCNN model manual<a name="run_basic_pcnn"></a>

Check out and run the [basic PCNN manual](https://github.com/MitchellvDoorn/MSc-Thesis/blob/main/code/basic_pcnn_manual.ipynb) in the [code](https://github.com/MitchellvDoorn/MSc-Thesis/tree/main/code) folder to see how to code of the basic PCNN model works.