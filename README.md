# Prototypical Networks for Few-shot Learning

Code for the NIPS 2017 paper [Prototypical Networks for Few-shot Learning](http://papers.nips.cc/paper/6996-prototypical-networks-for-few-shot-learning.pdf).

If you use this code, please cite our paper:

```
@inproceedings{snell2017prototypical,
  title={Prototypical Networks for Few-shot Learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
 }
 ```

## Training a prototypical network

### Install dependencies

* This code has been tested on Python 3.5. If you're using [conda](https://conda.io/docs/), you can create a Python 3.5 environment by running `conda create -n protonets python=3.5` and then activate it by running `source activate protonets`.
* Install [PyTorch and torchvision](http://pytorch.org/).
* Install torchnet and tqdm.
* Install the protonets package by running `python setup.py install` or `python setup.py develop`.

### Set up the Omniglot dataset

* Download [images_background.zip](https://github.com/brendenlake/omniglot/blob/master/python/images_background.zip?raw=true) and [images_evaluation.zip](https://github.com/brendenlake/omniglot/blob/master/python/images_evaluation.zip?raw=true).
* Extract both into `data/omniglot/data`.

### Train the model

* Run `python scripts/train/few_shot/run_train.py`. If you are running on a GPU you can pass in the option `--data.cuda`. This will run training and place the results into `results`.
  * You can specify a different output directory by passing in the option `--log.exp_dir EXP_DIR`, where `EXPD_DIR` is your desired output directory.
* Re-run in trainval mode `python scripts/train/few_shot/run_trainval.py`. This will save your model into `results/trainval` by default.

### Evaluate

* Run `python scripts/predict/few_shot/run_eval.py`.
