function[f_vec2,g_vec2,time_vec2,x] = Alg_Projection(fun_f,grad_f,grad_g,fun_g,param,x0)
% a-IRG in "A method with convergence rates for optimization
% problems with variational inequality constraints", H. D. Kaushik and F. Yousefian, SIOPT 2021
% The update rule is given by:
%
% x_{k+1} = x_k - \gamma_k(\nabla g(x_k)+\eta_k \nabla f(x_k))
%
% We choose \gamma_k = \gamma_0/sqrt{k+1} and \eta_k = \eta_0/(k+1)^0.25

eta_0 = 1e-3;
gamma_0 = 1/param.L_g;
f_vec2 = [];
g_vec2 = [];
time_vec2 = [];
x = x0;
% algorithm
maxiter = param.maxiter;
maxtime = param.maxtime;
tic;
for k = 1 : maxiter
    eta_k = (eta_0)/(k+1)^0.25;
    gamma_k = gamma_0/sqrt(k+1);
    % Descent step
    x = x - gamma_k*(grad_g(x)+eta_k*(grad_f(x)));
    cpu_t2 = toc;
    f_vec2 = [f_vec2;fun_f(x)];
    g_vec2 = [g_vec2;fun_g(x)];
    time_vec2 = [time_vec2;cpu_t2];
    if mod(k,5000) == 1
        fprintf('Iteration: %d\n',k)
    end
    if cpu_t2>maxtime
        break
    end
end
end