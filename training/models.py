from torch import nn
import torchvision.models as models
from torch.autograd import Variable


def set_parameter_requires_grad(model, number_frozen_layers, feature_layers=8):
    for k, child in enumerate(model.named_children()):
        if k == number_frozen_layers or k == feature_layers:
            break
        for param in child[1].parameters():
            param.requires_grad = False
    return model





def model_option(model_name: str,
                 num_classes: int,
                 freeze=False,
                 num_freezed_layers=0,
                 seg_mask=False,
                 dropout=0.0):
    """

    Parameters
    ----------
    model_name:str
    Named of the model to be initialized
    num_classes: int
    Number of classes that the classification model to be trained will have,
    this is the number of final output neurons
    freeze: bool
    Whether to freeze the weights of the pretrained network
    num_freezed_layers: int
    How many layers to freeze, if variable freeze is set to False this value is not important
    seg_mask: bool
    Whether the training is being done with 4-channeled images (RGB+segmentation mask)
    dropout: float
    Dropout rate

    Returns
    -------
    net
    Pretrained network with the specified characteristics
    resize_param
    resize parameter for the image that will be the input of the network
    """
    # if ever in need to delete cached weights go to Users\.cache\torch\hub\checkpoints (windows)
    net = None
    resize_param = 0
    if model_name.lower() == "resnet":
        """ ResNet50 """
        net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        if seg_mask:
            #   Modifying the input layer to receive 4-channel image instead of 3-channel image,
            #   We keep the pretrained weights for the RGB channels of the images
            weight1 = net.conv1.weight.clone()
            new_first_layer = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                        bias=False).requires_grad_()
            new_first_layer.weight[:, :3, :, :].data[...] = Variable(weight1, requires_grad=True)
            net.conv1 = new_first_layer

        if freeze:
            # Freezing the number of layers
            # ResNet has two named layers in position 2 and 3 named relu and maxpool
            # that do not have any learnable parameters, therefore if the user wants to
            # freeze more than 2 layers we offset the num_freezed_layers with two
            if num_freezed_layers > 2:
                net = set_parameter_requires_grad(net, num_freezed_layers + 2)
            else:
                net = set_parameter_requires_grad(net, num_freezed_layers)

        input_features = net.fc.in_features  # 2048
        net.fc = nn.Sequential(nn.Dropout(p=dropout),
                               nn.Linear(input_features, input_features // 4),
                               nn.ReLU(inplace=True),
                               nn.Dropout(p=dropout),
                               nn.Linear(input_features // 4, input_features // 8),
                               nn.ReLU(inplace=True),
                               nn.Dropout(p=dropout),
                               nn.Linear(input_features // 8, num_classes))
        resize_param = 224

    elif model_name.lower() == "convnext":
        """ ConvNeXt small """
        net = models.convnext_small(weights='DEFAULT')
        if seg_mask:
            #   Modifying the input layer to receive 4-channel image instead of 3-channel image,
            #   We keep the pretrained weights for the RGB channels of the images
            weight1 = net.features[0][0].weight.clone()
            bias1 = net.features[0][0].bias.clone()
            new_first_layer = nn.Conv2d(4, 96, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0),
                                        bias=True).requires_grad_()
            new_first_layer.weight[:, :3, :, :].data[...] = Variable(weight1, requires_grad=True)
            new_first_layer.bias.data[...] = Variable(bias1, requires_grad=True)
            net.features[0][0] = new_first_layer
        if freeze:
            # Freezing the number of layers
            net.features = set_parameter_requires_grad(net.features, num_freezed_layers)
        input_features = net.classifier[2].in_features  # 768
        net.classifier[2] = nn.Sequential(nn.Dropout(p=dropout),
                                          nn.Linear(input_features, input_features // 2),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=dropout),
                                          nn.Linear(input_features // 2, input_features // 4),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=dropout),
                                          nn.Linear(input_features // 4, num_classes))
        resize_param = 224

    elif model_name.lower() == "swin":
        """ Swin Transformer V2 -T """
        net = models.swin_v2_t(weights=models.Swin_V2_T_Weights.DEFAULT)
        if seg_mask:
            #   Modifying the input layer to receive 4-channel image instead of 3-channel image,
            #   We keep the pretrained weights for the RGB channels of the images
            weight1 = net.features[0][0].weight.clone()
            bias1 = net.features[0][0].bias.clone()
            new_first_layer = nn.Conv2d(4, 96, kernel_size=(4, 4), stride=(4, 4), padding=(0, 0),
                                        bias=True).requires_grad_()
            new_first_layer.weight[:, :3, :, :].data[...] = Variable(weight1, requires_grad=True)
            new_first_layer.bias.data[...] = Variable(bias1, requires_grad=True)
            net.features[0][0] = new_first_layer
        if freeze:
            # Freezing the number of layers
            net.features = set_parameter_requires_grad(net.features, num_freezed_layers)
        input_features = net.head.in_features  # 768
        net.head = nn.Sequential(nn.Dropout(p=dropout),
                                 nn.Linear(input_features, input_features // 2),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(input_features // 2, input_features // 4),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=dropout),
                                 nn.Linear(input_features // 4, num_classes))
        resize_param = 224

    elif model_name.lower() == "efficient":
        net = models.efficientnet_b0(weights='DEFAULT')
        if seg_mask:
            #   Modifying the input layer to receive 4-channel image instead of 3-channel image,
            #   We keep the pretrained weights for the RGB channels of the images
            weight1 = net.features[0][0].weight.clone()
            new_first_layer = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                        bias=False).requires_grad_()
            new_first_layer.weight[:, :3, :, :].data[...] = Variable(weight1, requires_grad=True)
            net.features[0][0] = new_first_layer
        if freeze:
            # Freezing the number of layers
            net.features = set_parameter_requires_grad(net.features, num_freezed_layers, feature_layers=9)
        input_features = net.classifier[1].in_features  # 1200
        net.classifier = nn.Sequential(nn.Dropout(p=dropout),
                                       nn.Linear(input_features, input_features // 2),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(p=dropout),
                                       nn.Linear(input_features // 2, input_features // 4),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(p=dropout),
                                       nn.Linear(input_features // 4, num_classes))
        resize_param = 224

    else:
        print("Invalid model name, exiting...")
        TypeError("Valid model names are 'resnet', 'convnext', 'swim' or 'efficient'")
        exit()

    return net, resize_param