
import torch
import torch.nn as nn
from torchvision.models.convnext import CNBlockConfig, ConvNeXt

config_map = {
    "convnext_tiny": {
        "url": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
        "depths": [3, 3, 9, 3],
        "dims": [96, 192, 384, 768],
    },
    "convnext_small": {
        "url": "https://download.pytorch.org/models/convnext_small-0c510722.pth",
        "depths": [3, 3, 27, 3],
        "dims": [96, 192, 384, 768],
    },
    "convnext_base": {
        "url": "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
        "depths": [3, 3, 27, 3],
        "dims": [128, 256, 512, 1024],
    },
    "convnext_large": {
        "url": "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
        "depths": [3, 3, 27, 3],
        "dims": [192, 384, 768, 1536],
    }
}

class MyConvNeXt(ConvNeXt):
    def _forward_impl(self, x):
        x = self.features(x)
        return x

class ConvNeXtLandmarkDetector(nn.Module):
    def __init__(self, configuration, num_landmarks, image_height=96, image_width=128, pretrained=True):
        super(ConvNeXtLandmarkDetector, self).__init__()
        config = config_map[configuration]
        self.pretrained = pretrained
        self.num_landmarks = num_landmarks
        self.image_height = image_height 
        self.image_width = image_width
        four_none = lambda x: x if x!=5 else None
        block_setting = [CNBlockConfig(config["dims"][i], (config["dims"][i+1] if i<3 else None), config["depths"][i]) for i in range(4)]
        self.backbone = MyConvNeXt(
            block_setting=block_setting, 
        )
        h_feat, w_feat = self.image_height//32, self.image_width//32
        emb_size = config["dims"][-1]
        low_emb_size = emb_size//9
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