# import models_mae
import torchvision
import torch
# from models_vit import vit_base_patch16
# from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np



if __name__ == '__main__':
    # x = torch.rand(16, 3, 224, 224)
    # y = torch.rand(16, 3, 224, 224)
    x1 = torch.arange(0,224//16,2)
    tmp = x1.clone()
    for x in x1:
        tmp += 224//16*x
        print(tmp)
    

    # checkpoint = torch.load('/Users/panzixuan/Documents/AI/checkpoints/checkpoint-60.pth', map_location='cpu')
    # model = checkpoint['model']
    # model0 = models_mae.MaskedAutoencoderViT()
    # optimizer = torch.optim.Adam(model0.parameters(), lr=0.05)
    # for i in range(1):
    #     criterion = torch.nn.CrossEntropyLoss()
    #     optimizer.zero_grad()
    #     total_loss, _, _, _, _ = model0(x, y, torch.zeros(1).long())
    #     total_loss.backward()
    #     optimizer.step()

