import torch
import torch.nn.functional as F

class MIL_model(torch.nn.Module):
    def __init__(self, model, hidden_space_len):

        super(MIL_model, self).__init__()
		
        self.model = model
        self.fc_input_features = self.model.input_features
        self.num_classes = self.model.num_classes
        self.hidden_space_len = hidden_space_len
        self.net = self.model.net

        self.conv_layers = torch.nn.Sequential(*list(self.net.children())[:-1])

        if (torch.cuda.device_count()>1):
            self.conv_layers = torch.nn.DataParallel(self.conv_layers)


        if self.model.embedding_bool:
            if ('resnet34' in self.model.model_name):
                self.E = self.hidden_space_len
                self.L = self.E
                self.D = self.hidden_space_len
                self.K = self.N_CLASSES

            elif ('resnet50' in self.model.model_name):
                self.E = 256
                self.L = self.E
                self.D = 256
                self.K = self.N_CLASSES

            self.embedding = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.E)
            self.post_embedding = torch.nn.Linear(in_features=self.E, out_features=self.E)

        else:
            self.fc = torch.nn.Linear(in_features=self.fc_feat_in, out_features=self.N_CLASSES)

            if ('resnet34' in self.model.model_name):
                self.L = self.fc_input_features
                self.D = self.hidden_space_len
                self.K = self.N_CLAS                
            elif ('resnet50' in self.model.model_name):
                self.L = self.E
                self.D = 256
                self.K = self.N_CLASSES
		
        if (self.model.pool_algorithm=="attention"):
            self.attention = torch.nn.Sequential(
                torch.nn.Linear(self.L, self.D),
                torch.nn.Tanh(),
                torch.nn.Linear(self.D, self.K)
            )
            self.embedding_before_fc = torch.nn.Linear(self.E * self.K, self.E)

        self.embedding_fc = torch.nn.Linear(self.E, self.K)

        self.dropout = torch.nn.Dropout(p=0.2)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()

        self.LayerNorm = torch.nn.LayerNorm(self.E * self.K, eps=1e-5)
        self.activation = self.tanh
        #self.activation = self.relu 

    def forward(self, x, conv_layers_out):

        #if used attention pooling
        A = None
        #m = torch.nn.Softmax(dim=1)
        m_binary = torch.nn.Sigmoid()
        m_multiclass = torch.nn.Softmax()
        dropout = torch.nn.Dropout(p=0.2)

        if x is not None:
            #print(x.shape)
            conv_layers_out=self.conv_layers(x)
            #print(x.shape)

            conv_layers_out = conv_layers_out.view(-1, self.fc_feat_in)

        n_patches = conv_layers_out.shape[0]

        #dk = max(1,np.log10(n_patches))
        #dk = max(1,np.log10(self.D))
        dk = 1

        if ('mobilenet' in self.model.model_name):
            conv_layers_out = self.dropout(conv_layers_out)


        if self.model.embedding_bool:
            embedding_layer = self.embedding(conv_layers_out)
							
            #embedding_layer = self.LayerNorm(embedding_layer)
            features_to_return = embedding_layer
            embedding_layer = self.dropout(embedding_layer)

        else:
            embedding_layer = conv_layers_out
            features_to_return = embedding_layer

        #print(features_to_return.size())
        #print("features_to_return: " + str(features_to_return))

        A = self.attention(features_to_return)

        A = torch.transpose(A, 1, 0)
        #print("A ante soft: " + str(A))
        A = F.softmax(A, dim=1)

        cls_img = torch.mm(A, features_to_return)
        cls_img = cls_img.view(-1, self.E * self.K)

        #cls_img = self.dropout(cls_img)

        cls_img = self.embedding_before_fc(cls_img)

        cls_img = self.activation(cls_img)
        feature_img = cls_img

        cls_img = self.dropout(cls_img)

        Y_prob = self.embedding_fc(cls_img)

        Y_prob = torch.squeeze(Y_prob)

        return Y_prob, feature_img	