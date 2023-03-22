from torch import nn
import torchvision.models as models


class ModelOption():
    def __init__(self, model_name: str,
                 num_classes: int,
                 freeze=False,
                 num_freezed_layers=0,
                 dropout=0.0, 
                 embedding_bool=False):

        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze = freeze
        self.num_freezed_layers = num_freezed_layers
        self.dropout = dropout
        self.embedding_bool = embedding_bool

        if self.model_name.lower() == "resnet50":
            """ ResNet50 """
            self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            if self.freeze:
                # Freezing the number of layers
                # ResNet has two named layers in position 2 and 3 named relu and maxpool
                # that do not have any learnable parameters, therefore if the user wants to
                # freeze more than 2 layers we offset the num_freezed_layers with two
                if self.num_freezed_layers > 2:
                    self.net = self.set_parameter_requires_grad(self.net,
                                                                self.num_freezed_layers + 2)
                else:
                    self.net = self.set_parameter_requires_grad(self.net,
                                                                
                                                                self.num_freezed_layers)

            self.input_features = self.net.fc.in_features  # 2048
            
            # self.net.fc = nn.Sequential(nn.Dropout(p=self.dropout),
            #                             nn.Linear(input_features,
            #                                       input_features // 4),
            #                             nn.ReLU(inplace=True),
            #                             nn.Dropout(p=self.dropout),
            #                             nn.Linear(input_features // 4,
            #                                      input_features // 8),
            #                             nn.ReLU(inplace=True),
            #                             nn.Dropout(p=self.dropout),
            #                             nn.Linear(input_features // 8,
            #                                      self.num_classes))

            self.resize_param = 224

        elif self.model_name.lower() == "convnext":
            """ ConvNeXt small """
            self.net = models.convnext_small(weights='DEFAULT')

            if self.freeze:
                # Freezing the number of layers
                self.net.features = self.set_parameter_requires_grad(self.net.features,
                                                                     self.num_freezed_layers)

            input_features = self.net.classifier[2].in_features  # 768
            self.net.classifier[2] = nn.Sequential(nn.Dropout(p=self.dropout),
                                                   nn.Linear(input_features,
                                                             input_features // 2),
                                                   nn.ReLU(inplace=True),
                                                   nn.Dropout(p=self.dropout),
                                                   nn.Linear(input_features // 2,
                                                             input_features // 4),
                                                   nn.ReLU(inplace=True),
                                                   nn.Dropout(p=self.dropout),
                                                   nn.Linear(input_features // 4,
                                                             self.num_classes))

            self.resize_param = 224

        elif self.model_name.lower() == "swin":
            """ Swin Transformer V2 -T """
            self.net = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)

            if self.freeze:
                # Freezing the number of layers
                self.net.features = self.set_parameter_requires_grad(self.net.features,
                                                                     self.num_freezed_layers)

            input_features = self.net.head.in_features  # 768
            self.net.head = nn.Sequential(nn.Dropout(p=self.dropout),
                                          nn.Linear(input_features,
                                                    input_features // 2),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=self.dropout),
                                          nn.Linear(input_features // 2,
                                                    input_features // 4),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=self.dropout),
                                          nn.Linear(input_features // 4,
                                                    self.num_classes))

            self.resize_param = 224

        elif self.model_name.lower() == "efficient":
            """ EfficientNet b0 """
            self.net = models.efficientnet_b0(weights='DEFAULT')

            if self.freeze:
                # Freezing the number of layers
                self.net.features = self.set_parameter_requires_grad(self.net.features,
                                                                     self.num_freezed_layers,
                                                                     feature_layers=9)

            input_features = self.net.classifier[1].in_features  # 1200
            self.net.classifier = nn.Sequential(nn.Dropout(p=self.dropout),
                                                nn.Linear(input_features,
                                                          input_features // 2),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=self.dropout),
                                                nn.Linear(input_features // 2,
                                                          input_features // 4),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=self.dropout),
                                                nn.Linear(input_features // 4,
                                                          self.num_classes))

            self.resize_param = 224

        else:
            print("Invalid model name, MODEL NOT LOAD")
            TypeError("Valid model names are 'resnet', 'convnext', 'swim' or 'efficient'")
            exit()

    def set_parameter_requires_grad(model, number_frozen_layers, feature_layers=8):
        for k, child in enumerate(model.named_children()):
            if k == number_frozen_layers or k == feature_layers:
                break
            for param in child[1].parameters():
                param.requires_grad = False

        return model
class ModelOption():
    def __init__(self, model_name: str,
                 num_classes: int,
                 freeze=False,
                 num_freezed_layers=0,
                 dropout=0.0, 
                 embedding_bool=False):

        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze = freeze
        self.num_freezed_layers = num_freezed_layers
        self.dropout = dropout
        self.embedding_bool = embedding_bool

        if self.model_name.lower() == "resnet50":
            """ ResNet50 """
            self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            if self.freeze:
                # Freezing the number of layers
                # ResNet has two named layers in position 2 and 3 named relu and maxpool
                # that do not have any learnable parameters, therefore if the user wants to
                # freeze more than 2 layers we offset the num_freezed_layers with two
                if self.num_freezed_layers > 2:
                    self.net = self.set_parameter_requires_grad(self.net,
                                                                self.num_freezed_layers + 2)
                else:
                    self.net = self.set_parameter_requires_grad(self.net,
                                                                
                                                                self.num_freezed_layers)

            self.input_features = self.net.fc.in_features  # 2048
            
            # self.net.fc = nn.Sequential(nn.Dropout(p=self.dropout),
            #                             nn.Linear(input_features,
            #                                       input_features // 4),
            #                             nn.ReLU(inplace=True),
            #                             nn.Dropout(p=self.dropout),
            #                             nn.Linear(input_features // 4,
            #                                      input_features // 8),
            #                             nn.ReLU(inplace=True),
            #                             nn.Dropout(p=self.dropout),
            #                             nn.Linear(input_features // 8,
            #                                      self.num_classes))

            self.resize_param = 224

        elif self.model_name.lower() == "convnext":
            """ ConvNeXt small """
            self.net = models.convnext_small(weights='DEFAULT')

            if self.freeze:
                # Freezing the number of layers
                self.net.features = self.set_parameter_requires_grad(self.net.features,
                                                                     self.num_freezed_layers)

            input_features = self.net.classifier[2].in_features  # 768
            self.net.classifier[2] = nn.Sequential(nn.Dropout(p=self.dropout),
                                                   nn.Linear(input_features,
                                                             input_features // 2),
                                                   nn.ReLU(inplace=True),
                                                   nn.Dropout(p=self.dropout),
                                                   nn.Linear(input_features // 2,
                                                             input_features // 4),
                                                   nn.ReLU(inplace=True),
                                                   nn.Dropout(p=self.dropout),
                                                   nn.Linear(input_features // 4,
                                                             self.num_classes))

            self.resize_param = 224

        elif self.model_name.lower() == "swin":
            """ Swin Transformer V2 -T """
            self.net = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)

            if self.freeze:
                # Freezing the number of layers
                self.net.features = self.set_parameter_requires_grad(self.net.features,
                                                                     self.num_freezed_layers)

            input_features = self.net.head.in_features  # 768
            self.net.head = nn.Sequential(nn.Dropout(p=self.dropout),
                                          nn.Linear(input_features,
                                                    input_features // 2),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=self.dropout),
                                          nn.Linear(input_features // 2,
                                                    input_features // 4),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=self.dropout),
                                          nn.Linear(input_features // 4,
                                                    self.num_classes))

            self.resize_param = 224

        elif self.model_name.lower() == "efficient":
            """ EfficientNet b0 """
            self.net = models.efficientnet_b0(weights='DEFAULT')

            if self.freeze:
                # Freezing the number of layers
                self.net.features = self.set_parameter_requires_grad(self.net.features,
                                                                     self.num_freezed_layers,
                                                                     feature_layers=9)

            input_features = self.net.classifier[1].in_features  # 1200
            self.net.classifier = nn.Sequential(nn.Dropout(p=self.dropout),
                                                nn.Linear(input_features,
                                                          input_features // 2),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=self.dropout),
                                                nn.Linear(input_features // 2,
                                                          input_features // 4),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=self.dropout),
                                                nn.Linear(input_features // 4,
                                                          self.num_classes))

            self.resize_param = 224

        else:
            print("Invalid model name, MODEL NOT LOAD")
            TypeError("Valid model names are 'resnet', 'convnext', 'swim' or 'efficient'")
            exit()

    def set_parameter_requires_grad(model, number_frozen_layers, feature_layers=8):
        for k, child in enumerate(model.named_children()):
            if k == number_frozen_layers or k == feature_layers:
                break
            for param in child[1].parameters():
                param.requires_grad = False

        return model