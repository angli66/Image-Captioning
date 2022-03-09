################################################################################
# CSE 151B: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Updated by Rohin
# Winter 2022
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def get_image_encoder():
    '''Return a pre-trained ResNet-50 image encoder for partial fine-tuning'''
    encoder = models.resnet50(pretrained=True)
    for param in encoder.parameters():
        param.requires_grad = False
    # encoder.fc = nn.Linear(2048, embedding_size)
    encoder.fc = nn.Identity()
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
        self.encoder = get_image_encoder()

        self.linear = nn.Linear(2048, embedding_size)

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)
        # self.rnn = nn.RNN(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)

    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_layers, batch_size, hidden_dim)
        """
        return (torch.zeros((2, batch_size, self.hidden_size)).cuda(), \
                torch.zeros((2, batch_size, self.hidden_size)).cuda())

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

        features = self.linear(features)

        features = features.unsqueeze(1)
        
        captions = captions[:, :-1]

        self.batch_size = features.shape[0]
        self.hidden = self.init_hidden(self.batch_size)

        # Embed the captions
        # Raise batch_size x sequence_length to batch_size x sequence_length x embedding_size
        embeddings = self.embedding(captions)

        # Concatenate image features and caption embeddings excluding <end>
        embeddings = torch.cat((features, embeddings), dim=1)
        outputs, self.hidden = self.lstm(embeddings, self.hidden)
        # outputs, self.hidden = self.rnn(embeddings, self.hidden)
        outputs = self.fc(outputs)
        
        return outputs

    def sample(self, images, max_length, temperature, deteministic=False):
        '''Sample captions (in form of word id) for given images.'''
        if deteministic == True:
            raise NotImplementedError("Deterministic generation not implemented")
        
        batch_size = images.shape[0]
        states = self.init_hidden(batch_size)
        sampled_ids = torch.zeros((batch_size, max_length))
        features = self.encoder(images)
        features = self.linear(features)
        inputs = features.unsqueeze(1)
        for i in range(max_length):
            # if i == 0:
            #     outputs, states = self.lstm(inputs)
            # else:
            #     outputs, states = self.lstm(inputs, states)

            outputs, states = self.lstm(inputs, states)

            outputs = self.fc(outputs)
            predictions = F.softmax(outputs / temperature, dim=-1)
            predictions = predictions.squeeze()
            predictions = torch.multinomial(predictions, 1).reshape(-1)
            sampled_ids[:, i] = predictions
            inputs = self.embedding(predictions)
            inputs = inputs.unsqueeze(1)
        
        return sampled_ids

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