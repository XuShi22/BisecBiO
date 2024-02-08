function [f,grad_f] = f_logistic(x)
% upper-level

f = x'*x/2;
grad_f = x;
end