################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################
import torch
import torch.nn as nn
import torchvision.models as models

# Return a pre-trained ResNet-50 image encoder for partial fine-tuning
def get_image_encoder(embedding_size):
    encoder = models.resnet50(pretrained=True)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.fc = nn.Linear(512, embedding_size)
    return encoder

class Decoder1(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=2):
        super(Decoder1, self).__init__()

        # Properties
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        # Layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, num_layers)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, features, captions):
        pass

# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    # You may add more parameters if you want

    raise NotImplementedError("Model Factory Not Implemented")