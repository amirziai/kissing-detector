import torch
from torch import nn
from torchvision import models


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def convnet_init(model_name: str,
                 num_classes: int,
                 feature_extract: bool,
                 use_pretrained: bool = True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    output_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.fc = nn.Identity()
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.classifier[6].in_features
        # model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        model_ft.classifier = nn.Identity()
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # num_ftrs = model_ft.classifier[6].in_features
        # model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        model_ft.fc = nn.Identity()
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # TODO: this is my attempt to remove the last FC layer, doesn't seem to work for SqueezeNet
        # model_ft.classifier = nn.Identity()
        # model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.classifier[1] = nn.Identity()
        # model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        # model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        model_ft.classifier = nn.Identity()
        input_size = 224

    # elif model_name == "inception":
    #     """ Inception v3
    #     Be careful, expects (299,299) sized images and has auxiliary output
    #     """
    #     model_ft = models.inception_v3(pretrained=use_pretrained)
    #     set_parameter_requires_grad(model_ft, feature_extract)
    #     # Handle the auxiliary net
    #     num_ftrs = model_ft.AuxLogits.fc.in_features
    #     model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    #     # Handle the primary net
    #     num_ftrs = model_ft.fc.in_features
    #     # model_ft.fc = nn.Linear(num_ftrs, num_classes)
    #     model_ft.fc = nn.Identity()
    #     input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    output_size = model_ft(torch.rand((1, 3, input_size, input_size))).shape[1]

    return model_ft, input_size, output_size
