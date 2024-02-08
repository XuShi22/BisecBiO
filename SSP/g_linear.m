function [c,g] = g_linear(x,A,b,g_star)
% lower-level

[m,n]=size(A);
g = [];
c = (A*x-b)'*(A*x-b)/2 - g_star;
end