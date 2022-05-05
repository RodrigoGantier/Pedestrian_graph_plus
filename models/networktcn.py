# from TorchSUL import Model as M 
import random
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

class ResBlock1D(nn.Module):
	def __init__(self, inch=1204, outchn=1024, k=3):
		super(ResBlock1D, self).__init__()

		self.k = k 
		self.c1 = nn.ModuleDict({ 
		'conv': nn.Conv1d(inch, outchn, kernel_size=k, stride=k, bias=False),
		'bn': nn.BatchNorm1d(outchn), 'act': nn.PReLU(outchn)})

		self.c2 = nn.ModuleDict({ 
		'conv': nn.Conv1d(inch, outchn, kernel_size=1, stride=1, bias=False),
		'bn': nn.BatchNorm1d(outchn), 'act': nn.PReLU(outchn)})


		# self.c1 = M.ConvLayer1D(k, outchn, stride=k, activation=M.PARAM_PRELU, batch_norm=True, usebias=False, pad='VALID')
		# self.c2 = M.ConvLayer1D(1, outchn, activation=M.PARAM_PRELU, batch_norm=True, usebias=False, pad='VALID')

	def forward(self, x):
		short = x

		branch = self.c1['conv'](x)
		branch = self.c1['bn'](branch)
		branch = self.c1['act'](branch)
		branch = F.dropout(branch, 0.5, self.training, False)
		branch = self.c2['conv'](branch)
		branch = self.c2['bn'](branch)
		branch = self.c2['act'](branch)
		branch = F.dropout(branch, 0.5, self.training, False)
		# slicing & shortcut
		# branch_shape = branch.shape[-1]
		# short_shape = short.shape[-1]
		# start = (short_shape - branch_shape) // 2
		short = short[:, :, self.k//2::self.k]
		res = short + branch
		# res = F.dropout(res, 0.25, self.training, False)

		return res

class Refine2dNet(nn.Module):
	def __init__(self, num_kpts, temp_length, input_dimension=3, output_dimension=3, output_pts=None):
		super(Refine2dNet, self).__init__()
	
		self.num_kpts = num_kpts
		self.temp_length = temp_length
		self.output_dimension = output_dimension
		self.output_pts = num_kpts if output_pts is None else output_pts
		self.input_dimension = input_dimension
		
		self.c1 = nn.ModuleDict({
			'conv': nn.Conv1d(self.output_pts * self.output_dimension, 1024, kernel_size=3, stride=3, bias=False),
			'bn': nn.BatchNorm1d(1024), 'act': nn.PReLU(1024)}
		)
		# self.c1 = M.ConvLayer1D(3, 1024, stride=3, activation=M.PARAM_PRELU, pad='VALID', batch_norm=True, usebias=False)
		
		self.r1 = ResBlock1D(inch=1024, outchn=1024, k=3)
		self.r2 = ResBlock1D(inch=1024, outchn=1024, k=3)
		self.r3 = ResBlock1D(inch=1024, outchn=1024, k=3)
		self.r4 = ResBlock1D(inch=1024, outchn=1024, k=3)
		# self.r3 = ResBlock1D(k=3, dilation=3)
		# self.c5 = M.ConvLayer1D(9, 256, activation=M.PARAM_PRELU, pad='VALID', batch_norm=True, usebias=False)
		self.c4 = nn.ModuleDict({
			'conv': nn.Conv1d(1024, self.output_pts * self.output_dimension, kernel_size=1, stride=1, bias=True)})
		# self.c4 = M.ConvLayer1D(1, self.output_pts * self.output_dimension)

	def forward(self, x, drop=True):
		x = x.view(x.shape[0], x.shape[1], self.num_kpts * self.input_dimension)
		x = x.permute(0,2,1)

		x = self.c1['conv'](x)
		x = self.c1['bn'](x)
		x = self.c1['act'](x)
		x = self.r1(x)
		x = self.r2(x)
		x = self.r3(x)
		x = self.r4(x)
		# x = self.r5(x)
		# x = self.c5(x)
		x = self.c4['conv'](x)
		x = x.permute(0, 2, 1)
		x = x.reshape(x.shape[0], x.shape[1], self.output_pts, self.output_dimension)
		return x 

	def evaluate(self, x):
		aa = []
		for i in range(x.shape[0]-self.temp_length+1):
			aa.append(x[i:i+self.temp_length])
		aa = torch.stack(aa, dim=0)
		y = self(aa)
		# y = y.permute(1,0,2,3)
		y = y.squeeze()
		return y

# class Discriminator2D(M.Model):
# 	def initialize(self):
# 		self.c1 = M.ConvLayer1D(1, 1024, activation=M.PARAM_PRELU)
# 		self.c2 = M.ConvLayer1D(1, 256, activation=M.PARAM_PRELU)
# 		self.c3 = M.ConvLayer1D(1, 256, activation=M.PARAM_PRELU)
# 		self.c4 = M.ConvLayer1D(1, 1)

# 	def forward(self, x):
# 		return self.c4(self.c3(self.c2(self.c1(x))))

