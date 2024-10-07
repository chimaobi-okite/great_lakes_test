import argparse
import os
import random
import json

import torch
import torch.nn as nn


def accuracy_ignore_padding(score, y_true, device, tag_pad_idx=0):
    """
    Computes accuracy while ignoring padding tokens.

    Params:
        score (torch.Tensor):
            The predicted scores (logits) of shape (batch_size, sequence_length, num_tags).
            It is the output from the model before applying Softmax.

        y_true (torch.Tensor):
            The ground truth tensor containing the true tag indices of shape (batch_size, sequence_length).
        
        device (torch.device):
            The device on which to perform the computation (CPU or CUDA).
        
        tag_pad_idx (int, optional):
            The index of the padding token in the tagset. Default is 0.
    
    Return:
        accuracy (torch.Tensor):
            The computed accuracy as a float tensor, ignoring the padding tokens.
    """
    # Get the predicted tag indices by taking the argmax over the last dimension (num_tags)
    y_pred = score.argmax(dim=1, keepdim=True)
    
    # Create a mask to identify non-padding elements in the ground truth
    mask_nonpad = (y_true != tag_pad_idx).nonzero(as_tuple=True)
    
    # Filter out padding tokens in y_true and y_pred based on the mask
    y_true = y_true[mask_nonpad]
    y_pred = y_pred[mask_nonpad].squeeze(1)  # Remove unnecessary dimension
    
    # Calculate the number of correct predictions
    correct = y_pred.eq(y_true)
    
    # Total number of non-padding tokens
    total = torch.tensor([y_true.shape[0]], dtype=torch.float32, device=device)
    
    # Compute and return accuracy
    return correct.sum().float() / total


class BiLstmTagger(nn.Module):
    """
    A class implementing a BiLSTM-based POS tagger using an LSTM for bidirectional sequence modeling and a linear layer for mapping
    hidden states to tag predictions.

    Params:
        input_dim (int):
            The size of the input, which is typically the vocabulary size when using nn.Embedding, or 0 if using GloVe.

        embedding_dim (int):
            The size of the input word embeddings, i.e., the dimensionality of the input vectors.

        hidden_dim (int):
            The size of the hidden layer in the LSTM, i.e., the dimensionality of the hidden states.

        tagset_size (int):
            The number of possible POS tags in the dataset, used to define the output layer size.

        n_layers (int, optional):
            The number of LSTM layers. Default is 2.

        bidirectional (bool, optional):
            If True, makes the LSTM bidirectional. Default is True.

        use_glove (bool, optional):
            A boolean flag indicating whether to use pre-trained GloVe embeddings or to learn embeddings using nn.Embedding.
            If True, the model expects pre-trained GloVe embeddings to be used. Default is True.

        dropout (float, optional):
            The dropout rate to use for regularization in the embedding, LSTM, and linear layers. Default is 0.25.

        pad_idx (int, optional):
            The index used for padding in nn.Embedding. Default is 0.

    Methods:
        forward(X):
            Forward pass of the model. Takes a batch of sentences or word indices as input, processes them through the embedding,
            LSTM, and linear layers, and returns the predicted tag scores for each word in the sentence.

            Params:
                X (torch.Tensor):
                    A tensor of shape (batch_size, sequence_length, embedding_dim) if using pre-trained embeddings (GloVe),
                    or (batch_size, sequence_length) if using nn.Embedding for word indices.

            Return:
                tag_scores (torch.Tensor):
                    A tensor of shape (batch_size, sequence_length, tagset_size) containing the predicted tag scores
                    (as probabilities) for each word.

        init_weights(m):
            Initializes the weights of the model with a normal distribution, useful for better model convergence.
    """

    def __init__(self,
                 input_dim,
                 embedding_dim,
                 hidden_dim,
                 tagset_size,
                 n_layers,
                 bidirectional,
                 use_glove,
                 dropout,
                 pad_idx=0):
        """
        Initialize the BiLSTM Tagger model with embedding, LSTM, and linear layers.
        """
        super(BiLstmTagger, self).__init__()

        self.use_glove = use_glove

        # Initialize embedding layer only if not using GloVe embeddings
        if not use_glove:
            # Embedding layer for converting word indices to embeddings
            self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)

        ######### TODO: Your code starts here #########

        # BiLSTM layer for sequence modeling
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)

        # Linear layer to map LSTM hidden states to tag predictions
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

        ######### TODO: Your code ends here #########

        # Apply custom weight initialization
        self.apply(self.init_weights)
        if not use_glove:
            self.embedding.weight.data[pad_idx] = torch.zeros(embedding_dim)

    def forward(self, inputs):
        """
        Forward pass of the BiLSTM Tagger model.

        Params:
            inputs (torch.Tensor):
                Input tensor of shape (batch_size, sequence_length). Each element is a word index if using nn.Embedding, or word embeddings if using GloVe.

        Return:
            tag_scores (torch.Tensor):
                A tensor of shape (batch_size, sequence_length, tagset_size) containing the predicted tag scores (as probabilities) for each word.
        """
        if not self.use_glove:
            # If not using GloVe, apply the embedding layer
            inputs = self.dropout(self.embedding(inputs))

        ######### TODO: Your code starts here #########

        tag_scores = self.hidden2tag(self.dropout(self.lstm(inputs)[0]))

        ######### TODO: Your code ends here #########

        return tag_scores

    @staticmethod
    def init_weights(m):
        """
        Initialize the weights of the model using a normal distribution.
        This helps with better model convergence during training.

        Params:
            m (nn.Module): The module for which the weights are initialized.
        """
        ######### TODO: Your code starts here #########
        if isinstance(m, nn.Linear):
            # Initialize weights with normal distribution
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        ######### TODO: Your code ends here #########


class RNNTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_size):
        super().__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = self.softmax(tag_space)
        return tag_scores


def train(training_file, vocab_and_tags_file, use_glove=True):

    assert os.path.isfile(training_file), "Training file does not exist"
    vocabulary, tag_labels = [], []

    # Select device based on if you have a GPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print('Using device:', device)

    # Your code starts here

    model = RNNTagger(1, 1, 10)  # Replace with code that actually trains a model
    model = model.to(device)

    # Your code ends here

    with open(vocab_and_tags_file, 'w') as json_file:
        json.dump({
            'vocabulary': vocabulary,
            'tags': tag_labels
        }, json_file, indent=4)  # indent=4 for pretty printing

    return model

def test(model_file, vocab_and_tags_file, test_file, label_file, use_glove=False):

    assert os.path.isfile(model_file), "Model file does not exist"
    # print(test_file)
    assert os.path.isfile(test_file), "Test file does not exist"
    assert os.path.isfile(label_file), "Label file does not exist"

    # Select device based on if you have a GPU
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    print('Using device:', device)

    # Load vocabulary and tags
    with open(vocab_and_tags_file, 'r') as json_file:
        vocab_and_tags = json.load(json_file)
    vocabulary = vocab_and_tags['vocabulary']
    tag_labels = vocab_and_tags['tags']

    # Your code starts here

    # Load model and put on GPU if you have one
    # model = RNNTagger(1, 1, 10)
    model = BiLstmTagger(
        input_dim=len(vocabulary) if not use_glove else None,
        embedding_dim=100,
        hidden_dim=128,  # Define HIDDEN_DIM appropriately
        tagset_size=len(tag_labels),
        n_layers=2,  # Set to the same number of layers as during training
        bidirectional=True,
        use_glove=use_glove,
        dropout=0.5  # Use the same dropout value as used during training
    )
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model = model.to(device)

    prediction = model(
        torch.rand(1000, 1, 1).to(device)
    )  # replace with inference from the loaded model

    ground_truth = torch.tensor([
        random.randint(0, 10) for _ in range(1000)
    ]).to(device)  # replace with actual labels from the data files

    score = accuracy_ignore_padding(prediction, ground_truth, device)
    score = score.item()

    # Your code ends here

    print(f"The accuracy of the model is {100*score:6.2f}%")


def main(params):
    """
    The main function that either trains or tests the model based on the passed parameters.

    If `params.train` is True, the model will be trained on the specified training file and saved to the model file.
    Otherwise, the model will be tested on the specified test dataset, and the accuracy score will be calculated.

    Params:
        params: 
            A set of parameters that control the training and testing behavior. It includes:
            
            - train (bool): 
                If True, the model is trained. Otherwise, the model is tested.
                
            - training_file (str):
                Path to the training data file, used when `train=True`.
                
            - vocab_and_tags_file (str):
                Path to the file containing vocabulary and POS tag information.
                
            - glove (bool):
                A flag indicating whether to use GloVe embeddings during training or testing.
                
            - model_file (str):
                Path where the model's state dictionary will be saved after training, or loaded for testing.
                
            - test_file (str, optional):
                Path to the test data file, used when `train=False`.
                
            - truth_file (str, optional):
                Path to the ground truth labels file, used when `train=False`.

    Return:
        None:
            The function does not return a value but saves the trained model or computes the accuracy score based on the mode.
    """
    
    if params.train:
        # Train the model using the provided training file, vocab, and GloVe settings
        model = train(
            params.training_file,
            params.vocab_and_tags_file,
            params.glove
        )
        
        # Move the model to the CPU and save its state dictionary
        model.cpu()
        torch.save(model.state_dict(), params.model_file)

    else:
        # Test the model on the test file and compute accuracy score
        score = test(
            params.model_file,
            params.vocab_and_tags_file,
            params.test_file,
            params.truth_file,
            # params.glove
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HMM POS Tagger")
    parser.add_argument("--train", action="store_const", const=True, default=False)
    parser.add_argument("--model_file", type=str, default="model.torch")
    parser.add_argument("--vocab_and_tags_file", type=str, default="vocab_and_tags.json")
    parser.add_argument("--training_file", type=str, default="")
    parser.add_argument("--test_file", type=str, default="")
    parser.add_argument("--truth_file", type=str, default="")

    main(parser.parse_args())
