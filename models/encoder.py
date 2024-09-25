
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    takes in the size of the embeded_vector to fed to rnn.
    this is not used for training but just get the feature vector of size embed size
    """
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True) #Load the model with all the pretrained weights
        for param in resnet.parameters():
            param.requires_grad_(False)
            #By setting requires_grad=False, you are telling PyTorch not to compute gradients for this tensor during backpropagation.
    
    # get all the layers except last as we are not intereseted in classification
        modules = list(resnet.children())[:-1] # last layer Linear(in_features=2048, out_features=1000, bias=True)
        self.resnet = nn.Sequential(*modules)#unpackign the layers
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1) # flatten the layer 
        features = self.embed(features)
        return features
    