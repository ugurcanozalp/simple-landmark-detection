
import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck 

config_map = {
    "resnet18": {
        "url": 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
        "layers": [2, 2, 2, 2],
        "width_per_group": 64, 
        "groups": 1, 
        "block_type": "basic",
    },
    "resnet34": {
        "url": 'https://download.pytorch.org/models/resnet34-b627a593.pth',
        "layers": [3, 4, 6, 3],
        "width_per_group": 64, 
        "groups": 1, 
        "block_type": "basic",
    },
    "resnet50": {
        "url": 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
        "layers": [3, 4, 6, 3],
        "width_per_group": 64, 
        "groups": 1, 
        "block_type": "bottleneck",
    },
    "resnet101": {
        "url": 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
        "layers": [3, 4, 23, 3],
        "width_per_group": 64, 
        "groups": 1, 
        "block_type": "bottleneck",
    },
    "resnet152": {
        "url": 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
        "layers": [3, 8, 36, 3],
        "width_per_group": 64, 
        "groups": 1, 
        "block_type": "bottleneck",
    },
    "resnext50_32x4d": {
        "url": 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
        "layers": [3, 4, 6, 3],
        "width_per_group": 4,
        "groups": 32, 
        "block_type": "bottleneck",
    },
    "resnext101_32x8d": {
        "url": 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
        "layers": [3, 4, 23, 3],
        "width_per_group": 8,
        "groups": 32, 
        "block_type": "bottleneck",
    },
    "wide_resnet50_2": {
        "url": 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
        "layers": [3, 4, 6, 3],
        "width_per_group": 128, 
        "groups": 1, 
        "block_type": "bottleneck",
    },
    "wide_resnet101_2": {
        "url": 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
        "layers": [3, 4, 23, 3],
        "width_per_group": 128,
        "groups": 1, 
        "block_type": "bottleneck",
    }
}

class MyResNet(ResNet):

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class ResNetLandmarkDetector(nn.Module):
    def __init__(self, configuration, num_landmarks, image_height=96, image_width=128, pretrained=True):
        super(ResNetLandmarkDetector, self).__init__()
        config = config_map[configuration]
        self.pretrained = pretrained
        self.image_height = image_height 
        self.image_width = image_width
        block = Bottleneck if config["block_type"]=="bottleneck" else BasicBlock
        self.num_landmarks = num_landmarks
        self.backbone = MyResNet(
            block = block,
            layers = config["layers"],
            groups = config["groups"], 
            width_per_group = config["width_per_group"],
        )
        h_feat, w_feat = self.image_height//32, self.image_width//32
        emb_size = 2048 if config["block_type"]=="bottleneck" else 512
        low_emb_size = emb_size//8
        feat_size = h_feat*w_feat*low_emb_size
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(emb_size, low_emb_size, 1),
            nn.Dropout(0.1), 
            nn.Flatten(),
            nn.Linear(feat_size, 3*num_landmarks),
            nn.Sigmoid()
        )
        #nn.init.constant_(self.head[-1].bias, 0.5)
        if self.pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(url=config["url"], map_location="cpu", check_hash=True)
            self.backbone.load_state_dict(checkpoint, strict=False)

    def forward(self, img):
        features = self.backbone(img)
        out = self.head(features)
        return out.reshape(-1, self.num_landmarks, 3)
