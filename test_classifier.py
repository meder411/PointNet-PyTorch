import torch
import torch.nn as nn
import torch.autograd as grad
import torch.nn.functional as F

import numpy as np

from torch.utils.data import DataLoader
from dataloader import ModelNet40

from models.pointnet_classifier import PointNetClassifier



def main():
	
	num_points = 2000
	dims = 3
	batch_size = 32
	dataset_root_path = '/data/ModelNet40/'
	model_path = 'classifier_model_state.pth'


	# Instantiate a dataset loader
	model_net = ModelNet40(dataset_root_path, test=True)
	data_loader = DataLoader(model_net, batch_size=batch_size,
		shuffle=False, num_workers=12)
	gt_key = model_net.get_gt_key()

	# Instantiate the network
	classifier = PointNetClassifier(num_points, dims).eval().cuda().double()
	classifier.load_state_dict(torch.load(model_path))

	# Keep track of the number of samples seen
	total_num_samples = 0
	class_num_samples = np.zeros(40)

	# Create length-40 arrays to track per class accuracy
	class_correct = np.zeros(40)
	class_incorrect = np.zeros(40)

	# Also keep track of total accuracy
	total_correct = 0
	total_incorrect = 0

	# Print some feedback
	print 'Starting evaluation...\n'
	print 'Processing {} samples in batches of {}...'.format(len(model_net), 
		batch_size)

	num_batches = len(model_net) / batch_size
	for i, sample in enumerate(data_loader):
		print 'Batch {} / {}'.format(i, num_batches)

		# Parse loaded data
		points = grad.Variable(sample[0]).cuda()
		target = grad.Variable(sample[1]).cuda()
		path = sample[2]

		# Forward pass
		pred, _ = classifier(points)

		# Update accuracy
		# print pred
		# print F.softmax(pred, dim=1)
		_, idx = torch.max(F.softmax(pred, dim=1), 1)

		idx = idx.cpu().numpy()
		target = target.cpu().numpy()
		total_num_samples += len(target)
		for j in xrange(len(target)):
			val = target[j]==idx[j]
			total_correct += val
			class_correct[target[j]] += val
			total_incorrect += np.logical_not(val)
			class_incorrect[target[j]] += np.logical_not(val)
			class_num_samples[target[j]] += 1


	print 'Done!'

	print 'Total Accuracy: {:2f}'.format(total_correct / 
		float(total_num_samples))
	print 'Per Class Accuracy:'
	for i in xrange(len(class_correct)):
		print '{}: {:2f}'.format(gt_key[i], 
			class_correct[i] / float(class_num_samples[i]))




if __name__ == '__main__':
	main()
