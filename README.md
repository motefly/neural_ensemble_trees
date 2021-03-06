## Neural Ensemble Trees

This repository is improved from [neural random forest](https://github.com/JohannesMaxWel/neural_random_forests). Thanks the original author a lot!

Relevant paper:

[Neural Random Forests](https://arxiv.org/abs/1604.07143).

Based on it, we add [LightGBM](https://github.com/Microsoft/LightGBM) model in the framework and impove the performance a lot.

## Requirements
This code is based on python3 and uses tensorflow 1.3.0.

First, let's make sure you have all packages needed:
```
pip3 install -r requirements.txt
```
Notice that the newest version (installed from github source code) of LightGBM is needed and can't installed by pip temporarily!


## Quick Start
For a quick start, let's download the [mpg](https://archive.ics.uci.edu/ml/datasets/auto+mpg) dataset from the UCI Machine Learning Repository (30KB):
```
cd datasets/data/mpg_data
sh download.sh
```

To run different Neural Random Forest models on the mpg dataset, execute this (takes ~2min) from the repository root directory:
```
python3 main.py mpg <randomforest or lightgbm>
```

## Other Datasets
To run the model on a new dataset, you must write a data loader function and add an option to `data_loader.py`.
For inspiration, check out the data loaders in `preprocessing/` which are for other datasets used in the [paper](https://arxiv.org/abs/1604.07143) . 

The data loader functions all return a pair _(X, Y)_, where _X_  is an input matrix of size `[# samples, # features]`, and _Y_  is a vector of regression outputs with size `[# samples]`.
