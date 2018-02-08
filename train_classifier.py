import torch
import torch.nn as nn
import torch.autograd as grad
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import time
import os.path as osp
import os


from dataloader import ModelNet40
from models.pointnet_classifier import PointNetClassifier


def main():
	

	num_points = 2000
	dims = 3
	batch_size = 32
	num_epochs = 60
	lr = 0.001
	printout = 20
	reg_weight = 0.001
	dataset_root_path = 'data/ModelNet40/'
	snapshot = 10
	snapshot_dir = 'snapshots'

	try:
		os.mkdir(snapshot_dir)
	except:
		pass


	# Instantiate a dataset loader
	model_net = ModelNet40(dataset_root_path)
	data_loader = DataLoader(model_net, batch_size=batch_size,
		shuffle=True, num_workers=12)
	gt_key = model_net.get_gt_key()

	# Instantiate the network
	classifier = PointNetClassifier(num_points, dims).train().cuda().double()
	loss = nn.CrossEntropyLoss()
	regularization = nn.MSELoss()
	optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
	scheduler = torch.optim.lr_scheduler.StepLR(
		optimizer, step_size=20, gamma=0.5)

	# Identity matrix for enforcing orthogonality of second transform
	identity = grad.Variable(torch.eye(64).double().cuda(), 
		requires_grad=False)

	# Some timers and a counter
	forward_time = 0.
	backprop_time = 0.
	network_time = 0.
	batch_counter = 0

	# Whether to save a snapshot
	save = False

	print 'Starting training...\n'

	# Run through all epochs
	for ep in xrange(num_epochs):

		if ep % snapshot == 0 and ep != 0:
			save = True

		# Update the optimizer according to the learning rate schedule
		scheduler.step()

		for i, sample in enumerate(data_loader):

			# Parse loaded data
			points = grad.Variable(sample[0]).cuda()
			target = grad.Variable(sample[1]).cuda()

			# Record starting time
			start_time = time.time()

			# Zero out the gradients
			optimizer.zero_grad()

			# Forward pass
			pred, T2 = classifier(points)

			# Compute forward pass time
			forward_finish = time.time()
			forward_time += forward_finish - start_time

			# Compute cross entropy loss
			pred_error = loss(pred, target)

			# Also enforce orthogonality in the embedded transform
			reg_error = regularization(
				torch.bmm(T2, T2.permute(0,2,1)), 
				identity.expand(T2.shape[0], -1, -1))

			# Total error is the weighted sum of the prediction error and the 
			# regularization error
			total_error = pred_error + reg_weight * reg_error

			# Backpropagate
			total_error.backward()

			# Update the weights
			optimizer.step()

			# Compute backprop time
			backprop_finish = time.time()
			backprop_time += backprop_finish - forward_finish

			# Compute network time
			network_finish = time.time()
			network_time += network_finish - start_time

			# Increment batch counter
			batch_counter += 1

			#------------------------------------------------------------------
			# Print feedback
			#------------------------------------------------------------------

			if (i+1) % printout == 0:
				# Print progress
				print 'Epoch {}/{}'.format(ep+1, num_epochs)
				print 'Batches {}-{}/{} (BS = {})'.format(i-printout+1, i,
					len(model_net) / batch_size, batch_size)
				print 'PointClouds Seen: {}'.format(
					ep * len(model_net) + (i+1) * batch_size)
				
				# Print network speed
				print '{:16}[ {:12}{:12} ]'.format('Total Time', 'Forward', 'Backprop')
				print '  {:<14.3f}[   {:<10.3f}  {:<10.3f} ]' \
					.format(network_time, forward_time, backprop_time)

				# Print current error
				print '{:16}[ {:12}{:12} ]'.format('Total Error', 
					'Pred Error', 'Reg Error')
				print '  {:<14.4f}[   {:<10.4f}  {:<10.4f} ]'.format(
					total_error.data[0], pred_error.data[0], reg_error.data[0])
				print '\n'

				# Reset timers
				forward_time = 0.
				backprop_time = 0.
				network_time = 0.

			if save:
				print 'Saving model snapshot...'
				save_model(classifier, snapshot_dir, ep)
				save = False


def save_model(model, snapshot_dir, ep):
	save_path = osp.join(snapshot_dir, 'snapshot{}.params' \
		.format(ep))
	torch.save(model.state_dict(), save_path)	



if __name__ == '__main__':
	main()
