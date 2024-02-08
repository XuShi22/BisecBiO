function [l,u,x_g,psi_g] = BisecBiO_initial(fun_f,fun_g,fun_h,grad_f,grad_g,x0,param)
% Initial bounds for l,u

disp('Initial bounds start');
proxh = @(x) x;
opts.maxit = param.maxiter;
opts.tol = param.epsilonf;
opts.verbose = true;
opts.L_f = param.L_f;
opts.L_g = param.L_g;
fun_h = @(x) 0;
% opts.L0 = param.L_f;
% [x_f, fval_f] = Greedy_FISTA(fun_g, fun_h, grad_g, proxh, x0, opts);
% l=max(fval_f-param.epsilonf,0.01);
l=0.01;

opts.maxit = param.maxiter;
opts.tol = param.tol;
opts.verbose = true;
opts.L0 = param.L_g;
proxh = @(x) ProjectOntoL1Ball(x,param.lam1);
[x_g, psi_g] = Greedy_FISTA(fun_g, fun_h, grad_g, proxh, x0, opts);
u=fun_f(x_g)+opts.L_f*param.lam1;

disp('Initial bounds are solved!');
end