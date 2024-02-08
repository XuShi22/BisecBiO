function [c,g] = g_logistic(x,A,b,g_star,epsilon_g,lambda_g)
% lower-level

[m,n]=size(A);
g = [];
c1 = (1/m)*ones(1,m)*log((1+exp(-(A*x).*b))) - g_star;
c2 = sum(abs(x))-lambda_g;
c = [c1;c2];
end