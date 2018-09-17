import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            #dropout=0.4,
                            batch_first=True)

        # the linear layer that maps the hidden state output dimension 
        # to the vocab_size
        self.hidden2vocab = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        # create embedded word vectors for each token in a batch of captions
        embeds = self.word_embeddings(captions[:,:-1])  # batch_size,cap_length -> batch_size,cap_length-1,embed_size

         # -> batch_size, caption (sequence) length, embed_size
        inputs = torch.cat((features.unsqueeze(dim=1),embeds), dim=1)

        lstm_out, _ = self.lstm(inputs);   # print (lstm_out.shape) -> batch_size, caplength, hidden_size

        # get the scores for the most likely words
        outputs = self.hidden2vocab(lstm_out);     # print (outputs.shape) -> batch_size, caplength, vocab_size
        
        return outputs  #[:,:-1,:] # discard the last output of each sample in the batch.

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
    
        caption = []
        
        # Don't frankly know why "states" is provided as an input parameter, it doesn't
        #     seem like the kind of thing a caller would care about.
        # initialize the hidden state and send it to the same device as the inputs
        hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device),
                  torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))

        # Now we feed the LSTM output and hidden states back into itself to get the caption
        for i in range(max_len):
            lstm_out, hidden = self.lstm(inputs, hidden) # batch_size=1, sequence length=1 ->1,1,embedsize
            outputs = self.hidden2vocab(lstm_out)        # 1,1,vocab_size
            outputs = outputs.squeeze(1)                 # 1,vocab_size
            wordid  = outputs.argmax(dim=1)              # 1
            caption.append(wordid.item())
            
            # prepare input for next iteration
            inputs = self.word_embeddings(wordid.unsqueeze(0))  # 1,1->1,1,embed_size
          
        return caption