from .resnet import ResNetLandmarkDetector
from .convnext import ConvNeXtLandmarkDetector

model_map = {
	"resnet": ResNetLandmarkDetector,
	"convnext": ConvNeXtLandmarkDetector
}