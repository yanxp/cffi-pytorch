int add_operator_cuda_forward(THCudaTensor *input1, THCudaTensor *input2,THCudaTensor *output);
int add_operator_cuda_backward(THCudaTensor *grad_output,THCudaTensor *grad_input);
