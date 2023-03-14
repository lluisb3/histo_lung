import torch


class Encoder(torch.nn.Module):
    def __init__(self, model, dim):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """

        super(Encoder, self).__init__()

        # pre_trained_network = torch.hub.load('pytorch/vision:v0.10.0', model, pretrained=True)

        # if (('resnet' in model) or ('resnext' in model)):
        # 	fc_input_features = model.fc.in_features
        # elif (('densenet' in model)):
        # 	fc_input_features = model.classifier.in_features
        # elif ('mobilenet' in model):
        # 	fc_input_features = model.classifier[1].in_features

        self.conv_layers = torch.nn.Sequential(*list(model.children())[:-1])

        if (torch.cuda.device_count() > 1):
            self.conv_layers = torch.nn.DataParallel(self.conv_layers)

        self.fc_input_features = model.input_features
        self.num_classes = model.num_classes

        self.dim = dim  # feature dimension

        if (EMBEDDING_bool == True):

            if ('resnet34' in model):
                self.E = self.dim
                self.L = self.E
                self.D = 64
                self.K = self.num_classes

            elif ('resnet50' in model):
                self.E = self.dim
                self.L = self.E
                self.D = 128
                self.K = self.num_classes

            elif ('resnet152' in model):
                self.E = self.dim
                self.L = self.E
                self.D = 128
                self.K = self.num_classes

            self.embedding = torch.nn.Linear(in_features=self.fc_input_features, out_features=self.E)
            self.post_embedding = torch.nn.Linear(in_features=self.E, out_features=self.E)

        self.prelu = torch.nn.PReLU(num_parameters=1, init=0.25) 
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        #if used attention pooling
        A = None
        #m = torch.nn.Softmax(dim=1)
        dropout = torch.nn.Dropout(p=0.2)
        relu = torch.nn.ReLU()
        tanh = torch.nn.Tanh()

        if x is not None:
            #print(x.shape)
            conv_layers_out = self.conv_layers(x)
            #print(x.shape)

            conv_layers_out = conv_layers_out.view(-1, self.fc_input_features)

        #print(conv_layers_out.shape)

        # if ('mobilenet' in model):
        # 	#dropout = torch.nn.Dropout(p=0.2)
        # 	conv_layers_out = dropout(conv_layers_out)
        # #print(conv_layers_out.shape)

        if (EMBEDDING_bool == True):
            #embedding_layer = self.relu(conv_layers_out)
            embedding_layer = self.embedding(conv_layers_out)
            embedding_layer = self.relu(embedding_layer)
            embedding_layer = self.post_embedding(embedding_layer)

            features_to_return = embedding_layer

        else:
            features_to_return = conv_layers_out

        norm = torch.norm(features_to_return, p='fro', dim=1, keepdim=True)

        #normalized_array = features_to_return #/ norm
        #normalized_array = torch.nn.functional.normalize(features_to_return, dim=1)
        #normalized_array = torch.norm(features_to_return, p='fro', dim=1, keepdim=True)
        normalized_array = features_to_return 

        return normalized_array
    