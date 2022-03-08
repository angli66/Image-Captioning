################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nltk
from datetime import datetime

import caption_utils
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        # Set Criterion and Optimizers
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = optim.Adam(self.__model.parameters(), lr=config_data['experiment']['learning_rate'])

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()
        self.__best_model = self.__model

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            print("Training...")
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            self.__save_best_model()

        self.plot_stats()

    # Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        self.__model.train()
        training_loss = 0

        # Iterate over the data, implement the training function
        for images, captions, _ in self.__train_loader:
            images = images.cuda()
            captions = captions.cuda()

            self.__optimizer.zero_grad()
            outputs = self.__model(images, captions)
            targets = (F.one_hot(captions, num_classes=len(self.__vocab))).float()
            loss = self.__criterion(outputs, targets)
            loss.backward()
            self.__optimizer.step()
            training_loss += loss.item()
        
        training_loss /= len(self.__train_loader)
        return training_loss

    # Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = 0

        with torch.no_grad():
            for images, captions, _ in self.__val_loader:
                images = images.cuda()
                captions = captions.cuda()

                outputs = self.__model(images, captions)
                targets = (F.one_hot(captions, num_classes=len(self.__vocab))).float()
                loss = self.__criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(self.__val_loader)
        if len(self.__val_losses) == 0 or val_loss < min(self.__val_losses):
            self.__best_model = self.__model

        return val_loss

    #  Generate sample captions and evaluate loss and bleu scores using the best model.
    #  Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__model.eval()
        test_loss = 0
        bleu1 = 0
        bleu4 = 0

        with torch.no_grad():
            for images, captions, img_ids in self.__test_loader:
                images = images.cuda()
                captions = captions.cuda()

                outputs = self.__best_model(images, captions)

                # Calculate test loss
                targets = (F.one_hot(captions, num_classes=len(self.__vocab))).float()
                loss = self.__criterion(outputs, targets)
                test_loss += loss.item()

                # Generate the output string
                sampled_ids = self.__best_model.sample(images,
                                                        self.__generation_config['max_length'],
                                                        self.__generation_config['temperature'],
                                                        self.__generation_config['deterministic'])
                for i, sampled_id in enumerate(sampled_ids):
                    # Get predicted caption
                    predicted_caption = ""
                    for j, word_idx in enumerate(sampled_id):
                        if j == 0:
                            continue
                        word = self.__vocab.idx2word[word_idx.item()]
                        if word == '<end>':
                            break
                        predicted_caption += word + ' '
                    predicted_caption = nltk.tokenize.word_tokenize(predicted_caption)

                    # Get reference captions
                    reference_captions = []
                    annotations = self.__coco_test.imgToAnns[img_ids[i]]
                    for annotation in annotations:
                        reference_caption = annotation['caption']
                        reference_caption = nltk.tokenize.word_tokenize(reference_caption)
                        reference_captions.append(reference_caption)
                    
                    bleu1 += caption_utils.bleu1(reference_captions, predicted_caption)
                    bleu4 += caption_utils.bleu4(reference_captions, predicted_caption)
        
        test_loss /= len(self.__test_loader)
        bleu1 /= len(self.__test_loader.dataset)
        bleu4 /= len(self.__test_loader.dataset)
        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}".format(test_loss,
                                                                                bleu1,
                                                                                bleu4)
        self.__log(result_str)

        return test_loss, bleu1, bleu4

    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __save_best_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'best_model.pt')
        model_dict = self.__best_model.state_dict()
        torch.save(model_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
