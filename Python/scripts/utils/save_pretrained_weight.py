import torch
import torchvision as torchvision


def save_pretrained_weight():
    #model = torchvision.models.segmentation.fcn_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None)
    model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None)
    state_dict = model.state_dict()
    del state_dict['classifier.4.weight']
    del state_dict['classifier.4.bias']
    # aux_keys = []
    # for k in state_dict.keys():
    #     if "aux_classifier" in k:
    #         aux_keys.append(k)
    # for k in aux_keys:
    #     del state_dict[k]

    torch.save(state_dict, "/home/teo/storage/Data/pretrained_weight_DeepLab101")


if __name__ == '__main__':
    save_pretrained_weight()

