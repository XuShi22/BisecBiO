function [f,grad_f] = f_linear(x,alpha)
% upper-level

f = sum(abs(x)) + alpha*(x'*x)/2;
grad_f = [];
end