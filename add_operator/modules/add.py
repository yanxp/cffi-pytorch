from torch.nn.modules.module import Module
from functions.add import AddFunction
class AddModule(Module):
	def forward(self,input1,input2):
		return AddFunction()(input1,input2)