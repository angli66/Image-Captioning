################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################
import torch
import torch.nn as nn
import torchvision.models as models

def get_image_encoder(embedding_size):
    '''Return a pre-trained ResNet-50 image encoder for partial fine-tuning'''
    encoder = models.resnet50(pretrained=True)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.fc = nn.Linear(2048, embedding_size)
    return encoder

class Model1(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=2):
        super(Model1, self).__init__()

        # Properties
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Layers
        self.encoder = get_image_encoder(embedding_size)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, images, captions):
        '''
        Forward pass of the model.

        Parameters
        ----------
        images : tensor
            batch_size x height x width tensor that stores information of the images
        captions : tensor
            batch_size x sequence_length tensor that stores the index of the word in vocab

        Returns
        -------
        tensor
            batch_size x sequence_length x vocab_size tensor contains prediction
        '''
        # Get embedded features of images
        features = self.encoder(images)
        features = features.unsqueeze(1)

        # Embed the captions
        # Raise batch_size x sequence_length to batch_size x sequence_length x embedding_size
        embeddings = self.embedding(captions)

        # Concatenate image features and caption embeddings excluding <end>
        embeddings = torch.cat((features, embeddings[:, :-1, :]), dim=1)
        outputs, _ = self.lstm(embeddings)
        outputs = self.fc(outputs)
        
        return outputs

def get_model(config_data, vocab):
    '''Build and return the model here based on the configuration.'''
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    if model_type == 'model_1':
        model = Model1(len(vocab), embedding_size, hidden_size, num_layers=2)
        return model
    else:
        raise NotImplementedError(f"model {model_type} not implemented")