function [f_vec0,g_vec0,time_vec0,x_hat] = BisecBiO(fun_f,fun_g,grad_g,l,u,x_g,psi_g,param)
% Bisec-BiO method

opts.maxit = param.maxiter;
opts.tol = param.tol;
opts.verbose = true;
opts.L0 = param.L0;
opts.L_f = param.L_f;
opts.L_g = param.L_g;
f_vec0 = [];
g_vec0 = [];
time_vec0 = [];
c = l;
x_0c = x_g;
tic;
prox_phi = @(x) ProjectOntoL2Ball(x, sqrt(2*c));
fun_c = @(x) 0;
[x_c,g_c] = Greedy_FISTA(fun_g, fun_c, grad_g, prox_phi, x_0c, opts);
f_vec0 = [f_vec0;fun_f(x_c)];
g_vec0 = [g_vec0;g_c];
cpu_t0 = toc;
time_vec0 = [time_vec0;cpu_t0];
if  (g_c-psi_g-param.epsilong) <=0 % l is not lower bound (error)
    x_hat = x_c;
    return
else
    while u - l > 2*param.epsilonf
        c = (l + u) / 2;
        x_0c=x_c;
        prox_phi = @(x) ProjectOntoL2Ball(x, sqrt(2*c));
        [x_c,g_c] = Greedy_FISTA(fun_g, fun_c, grad_g, prox_phi, x_0c, opts);
        f_vec0 = [f_vec0;fun_f(x_c)];
        g_vec0 = [g_vec0;g_c];
        cpu_t0 = toc;
        time_vec0 = [time_vec0;cpu_t0];
        if (g_c-psi_g-param.epsilong) <=0
            u = c;
        else
            l = c;
        end
    end
end

c = u;
prox_phi = @(x) ProjectOntoL2Ball(x, sqrt(2*c));
x_0c = x_c;
[x_c,g_c] = Greedy_FISTA(fun_g, fun_c, grad_g, prox_phi, x_0c, opts);
f_vec0 = [f_vec0;fun_f(x_c)];
g_vec0 = [g_vec0;g_c];
cpu_t0 = toc;
time_vec0 = [time_vec0;cpu_t0];
x_hat = x_c;
end