import imageio
import numpy as np
import pprint
from skimage.transform import resize
import onnxruntime as ort

class LandmarkDetectionInference:

	def __init__(self, onnx_path, image_height, image_width):
		self.ort_session = ort.InferenceSession(onnx_path)
		self.image_height, self.image_width = image_height, image_width

	def __call__(self, image):
		image = self.preprocess(image)
		image_input = np.expand_dims(image.transpose(2,0,1), 0)
		# compute ONNX Runtime output prediction
		ort_inputs = {'image': image_input}
		ort_outs = self.ort_session.run(None, ort_inputs)
		output = ort_outs[0][0]
		return output, image

	def preprocess(self, image, render=False):
		image = resize(image, (self.image_height, self.image_width))
		if image.dtype == "uint8":
			image = image/255.0
		if len(image.shape) == 2:
			image = np.expand_dims(image, axis=-1).repeat(3, -1)
		return np.float32(image)

	def infer_visualize(self, image):
		output_numpy, image = self(image)
		import matplotlib.pyplot as plt 
		plt.imshow(image)
		x, y, visible = self.image_width*output_numpy[:,0], self.image_height*output_numpy[:,1], output_numpy[:,2]
		for i, vis in enumerate(visible):
			text = str(i) #if vis>0.5 else "x"
			symbol = "o" if vis>0.5 else "x"
			plt.annotate(symbol, (x[i], y[i]), color='green', fontsize='large')
			plt.annotate(text, (x[i], y[i]), color='red', fontsize='large')

		plt.show()

if __name__ == '__main__':
	model = LandmarkDetectionInference('deployment/convnext_tiny/convnext_tiny-quantized.onnx', 96, 128)
	image = imageio.imread("data/sample_cow.jpg")
	import time 
	t0 = time.perf_counter()
	for i in range(10):
	    result = model(image)
	dt = time.perf_counter() - t0
	#pprint.pprint(result)
	print(f'Elapsed time: {dt} seconds..')
	model.infer_visualize(image)


