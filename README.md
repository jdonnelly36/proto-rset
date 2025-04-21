# Proto-RSet

This is the official repository for "Rashomon Sets for Prototypical-Part Models: Editing Accurate Interpretable Models in Real-Time", published in CVPR 2025. This repository is built on the code from the arXiv paper "This Looks Better than That: Better Interpretable Models with ProtoPNeXt," and as such we would like to particularly acknowledge the code contributions of Frank Willard, Luke Moffett, Emmanuel Mokel, Julia Yang, and Giyoung Kim to this repository. Additionally, we include the code repository for "Concept-level Debugging of Part-Prototype Networks" to support replication of our experiments.

If you want to immediately try a Proto-RSet out for yourself, see `quickstart.ipynb`. This notebook is setup to run end to end in Google collab, with GPU enabled.

If you would like to test Proto-RSet on your own dataset, you will need to prepare your dataset and train a reference ProtoPNet. Please refer to the `Dev Environment Configuration` section for instructions to configure your environment and the `Training Your First ProtoPNet` section for an overview of how to train a ProtoPNet. See `Custom Dataset Preparation` for instructions on preparing a dataset for use in this codebase.

All experimental code for the CVPR paper is provided in the `experiments` directory, and the notebooks used to generate figures from those results are in `notebooks`.

This project is built using `pytorch`.

## Overview

Proto-RSet is a simple framework that enables users to quickly constrain prototype based models. Behind the scenes, we compute a set of many near-optimal alternative last layers for a given ProtoPNet, which allows us to directly manage which prototypes are/are not used by selecting different models. This enables users to produce a customized interpretable neural network in a matter of minutes. 

### Overview of ProtoPNets

ProtoPNet is a neural network framework that uses “prototypical examples” or "prototypes" from the training data in order to create predictions that are easily interpreted by humans. By using these prototypes, the model is able to classify a given image by explicitly connecting some part or feature of the image to some other example it has seen during training, allowing humans to view and validate the model's decision-making process. This is a favorable alternative to post-hoc interpretation of models, which relies on attempting to find images that result in high activation after the model has actually been created. Rather than finding images that result in high activation for a trained model, ProtoPNet actually incorporates a human-like reasoning process directly into its architecture.

The model gets prototypes from the feature space of a CNN without the final classification layer. This codebase provides backbones for Resnet, Densenet, VGG, and SparcNet. The prototypes are learned in training, then pushed onto an existing training image to allow for interpretability. This version of ProtoPNet has some optional extended functionality from two other related papers. One such addition is a deformable version of ProtoPNet, which allows components of each prototype to be displaced around an image and therefore create more robust prediction capabilities. We also added functionality to use finely annotated data, which forces the model to learn prototypes in areas of the image that we know to be relevant. A model is trained by running main.py. The model parameters can be passed in via the command line or a .yaml file. For more information about model parameters, see `./SETTINGS.md`.

### Papers

This codebase implements functionality from the following papers:
- "IAIA-BL: A Case-based Interpretable Deep Learning Model for Classification of Mass Lesions in Digital Mammography". Alina Jade Barnett, Fides Regina Schwartz, Chaofan Tao, Chaofan Chen, Yinhao Ren, Joseph Y. Lo and Cynthia Rudin
- "Deformable ProtoPNet: An Interpretable Image Classifier Using Deformable Prototypes".Jon Donnelly, Alina Jade Barnett, and Chaofan Chen
- "This Looks Like That: Deep Learning for Interpretable Image Recognition", authored by Chaofan Chen, Oscar Li, Chaofan Tao, Alina Jade Barnett, Jonathan Su, and Cynthia Rudin
- "Evaluation and improvement of interpretability for self-explainable part-prototype networks." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023. Huang, Qihan, et al.

## Dev Environment Configuration

This project is built on python 3.10.
Then upgrade pip to the latest version:
```pip install --upgrade pip```

Check you're running python 3.10 with

```
interpnn2023> python --version
Python 3.10.12
```

### ProtoPNet Isolated Environment

First, set up an isolated `virtualenvironment` or `conda` environment for this project.
If you choose to use `conda`, the dependency files are still written using pip addresses,
so you will still need to manage dependencies with pip.

#### Option 1: Virtual Environment

Make sure you have `virtualenv` installed:

```
pip install virtualenv
```

Then create a new virtual environment in the project root directory:
```virtualenv .venv```

Then activate the virtual environment.
On Unix/Mac:
```source .venv/bin/activate```
On Windows:
```.venv\Scripts\activate```

#### Option 2: Conda

You must have a `conda` installation.
If you do not have one, consider `miniconda`, which can be run directly from the executable on linux environments.

This can be done with the following commands:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

You should now be able to activate the project with `conda activate protopnext`

### Install Dependencies

To run with the exact versions used in our experiments, run:

```{sh}
pip install -r env/requirements.txt
```

Note that you may have trouble installing these exact versions with more recent versions of python.
If this is the case, you may try installing the required packages without specified verisons:

```{sh}
pip install -r env/requirements-collab.txt
```

## Training Your First ProtoPNet

This will walk you through the commands to setup a CUB-200 dataset and train ProtoPNet.

### Custom Dataset Prep

To train a reference ProtoPNet (and a prepare a Proto-RSet), we need an image datase structured to cooperate with this codebase. For standard image datasets, we assume all images are downloaded into a directory structured as follows:

```
| DATA_DIR
|-- |images
|-- |-- CLASS1
|-- |-- |-- class1_img1.jpg
|-- |-- |-- class1_img2.jpg
|-- |-- |-- ...
|-- |-- CLASS2
|-- |-- |-- class2_img1.jpg
|-- |-- |-- class2_img2.jpg
|-- |-- |-- ...
|-- |-- ...
```

You may manually replicate this structure in `train`, `val`, and `test` directories, or use a provided utility to generate these splits by calling `python -m protopnet create-splits DATA_DIR`. Finally, record the path to this processed dataset with `DATASET_DIR=/path/to/your/dataset;DATASET_NAME=generic_dataset`.

#### Advanced Dataset Setup

If your dataset requires specialized loading/preprocessing, you may need to write a custom loader for your dataset. You may do so by writing a new file in `protopnet/datasets` called `<DATASET_NAME>.py` (replacing DATASET_NAME with the name of your dataset). This file should contain a `train_dataloaders` function which returns a `LoaderBundle` object. 

### CUB-200 Dataset Prep (Skip if Using Your Own Dataset)

1. Download the dataset CUB_200_2011.tgz from https://www.vision.caltech.edu/datasets/cub_200_2011/ using the command `wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz`
2. Unpack CUB_200_2011.tgz. 
Check to see if the file unpacked properly
3. Run `python -m protopnet create-splits /path/to/unpacked/CUB_200_2011`. This will create `train`, `validation`, and `test` directories in your `CUB_200_2011` directory that match the splits used for ProtoPNeXt.
4. Set your CUB_200 directory to the variable `export CUB200_DIR=/path/to/your/dataset/CUB_200_2011_2/;DATASET_DIR=/path/to/your/dataset/CUB_200_2011_2/;DATASET_NAME=cub200`

### Training Your first ProtoPNet

On the isolated environment you set up with CUDA available, run the following:

1. Log in to the weights and biases with `wandb login`
2. Run `python -m protopnet train-vanilla-cos --dataset=$DATASET_NAME --dataset-dir=$DATASET_DIR`. See `protopnet.train_vanilla_cosine.main` for the default parameters. (Suggestion: It might be worthwhile to change the `phase_multiplier` parameter value to something small like 1-5 for your first run of ProtoPNet to reduce the number of epochs in each training phase, but this is not necessary).

Local logging shows the progress of the training in Epochs.
In Weights and Biases, you can view the training curves.

The CLI does not currently support an easy way to modify hyperparameters for single runs outside of a Weights and Biases Sweep.

### Running a Sweep

To do a full hyperparameter sweep on a model, see `training/README.md`.