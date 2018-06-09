import torch
from torch.autograd import Function
from _ext import add_operator
class AddFunction(Function):
	"""docstring for AddFunction"""
	def forward(self,input1,input2):
		output = input1.new()
		if not input1.is_cuda:
			add_operator.add_operator_forward(input1,input2,output)
		else:
			add_operator.add_operator_cuda_forward(input1,input2,output)
		return output
	def backward(self,grad_out):
		grad_int = grad_out.new()
		if not grad_out.is_cuda:
			add_operator.add_operator_backward(grad_out,grad_int)
		else:
			add_operator.add_operator_cuda_backward(grad_out,grad_int)
		return grad_int
