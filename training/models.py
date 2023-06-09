from torch import nn
import torchvision.models as models


class ModelOption():
    def __init__(self, model_name: str,
                 num_classes: int,
                 freeze=False,
                 num_freezed_layers=0,
                 dropout=0.0, 
                 embedding_bool=False,
                 pool_algorithm=None
                 ):

        self.model_name = model_name
        self.num_classes = num_classes
        self.num_freezed_layers = num_freezed_layers
        self.dropout = dropout
        self.embedding_bool = embedding_bool
        self.pool_algorithm = pool_algorithm

        if self.model_name.lower() == "resnet50":
            """ ResNet50 """
            self.net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

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

        elif self.model_name.lower() == "resnet34":
            """ ResNet34 """
            self.net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

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

        elif self.model_name.lower() == "resnet101":
            """ ResNet101 """
            self.net = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

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


            self.input_features = self.net.classifier[2].in_features  # 768
            # self.net.classifier[2] = nn.Sequential(nn.Dropout(p=self.dropout),
            #                                        nn.Linear(input_features,
            #                                                  input_features // 2),
            #                                        nn.ReLU(inplace=True),
            #                                        nn.Dropout(p=self.dropout),
            #                                        nn.Linear(input_features // 2,
            #                                                  input_features // 4),
            #                                        nn.ReLU(inplace=True),
            #                                        nn.Dropout(p=self.dropout),
            #                                        nn.Linear(input_features // 4,
            #                                                  self.num_classes))

            self.resize_param = 224

        elif self.model_name.lower() == "swin":
            """ Swin Transformer V2 -T """
            self.net = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)

            self.input_features = self.net.head.in_features  # 768
            # self.net.head = nn.Sequential(nn.Dropout(p=self.dropout),
            #                               nn.Linear(input_features,
            #                                         input_features // 2),
            #                               nn.ReLU(inplace=True),
            #                               nn.Dropout(p=self.dropout),
            #                               nn.Linear(input_features // 2,
            #                                         input_features // 4),
            #                               nn.ReLU(inplace=True),
            #                               nn.Dropout(p=self.dropout),
            #                               nn.Linear(input_features // 4,
            #                                         self.num_classes))

            self.resize_param = 224

        elif self.model_name.lower() == "efficient":
            """ EfficientNet b0 """
            self.net = models.efficientnet_b0(weights='DEFAULT')

            self.input_features = self.net.classifier[1].in_features  # 1200
            # self.net.classifier = nn.Sequential(nn.Dropout(p=self.dropout),
            #                                     nn.Linear(input_features,
            #                                               input_features // 2),
            #                                     nn.ReLU(inplace=True),
            #                                     nn.Dropout(p=self.dropout),
            #                                     nn.Linear(input_features // 2,
            #                                               input_features // 4),
            #                                     nn.ReLU(inplace=True),
            #                                     nn.Dropout(p=self.dropout),
            #                                     nn.Linear(input_features // 4,
            #                                               self.num_classes))

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
