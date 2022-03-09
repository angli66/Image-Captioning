[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=7169743&assignment_repo_type=AssignmentRepo)

# Image Captioning

This repository contains the code for a neural network that is trained to automatically generate captions for images. To run the code, first specify the hyperparameters of the system. Check `baseline.json` for description of the hyperparameters. Run with

    python3 main.py

will load `baseline.json` by default. After the training finished, it will generate a folder named `experiment_data` in the root folder that contains training information along with the test performance report (measures by BLEU score comparing to groundtruth captions). Then, run `visualize_results.ipynb` to visualize the generated captions by the model on the images from test set. To run with different config, create a new json file e.g. `new_experiment.json`, then run with

    python3 main.py new_experiment

## Usage

* Define the configuration for your experiment. See `default.json` to see the structure and available options. You are free to modify and restructure the configuration as per your needs.
* Implement factories to return project specific models, datasets based on config. Add more flags as per requirement in the config.
* Implement `experiment.py` based on the project requirements.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir. This can be configured in `contants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training pr evaluate performance.

## Files
- main.py: Main driver class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- dataset_factory: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 
- caption_utils.py: utility functions to generate bleu scores
- vocab.py: A simple Vocabulary wrapper
- coco_dataset: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- get_datasets.ipynb: A helper notebook to set up the dataset in your workspace
