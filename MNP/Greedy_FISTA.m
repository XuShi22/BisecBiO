function [x, fk] = Greedy_FISTA(fun_f, fun_g, grad_f,proxg, x0, opts)
% Greddy FISTA (Algorithm 4.3) in "Improving "fast iterative shrinkage-thresholding algorithm": faster, smarter, and greedier",
% J. Liang, T. Luo, and CB. Schonlieb, SIAM J Sci Comput 2022
% Improve the performance of restarting strategy by greedy parameter choices
% Solve subproblem

if ~isfield(opts, 'maxit')
    opts.maxit = 3e+5;
end
if ~isfield(opts, 'c_gamma')
    opts.c_gamma = 1.3;
end
if ~isfield(opts, 'a')
    opts.a = @(k) 1; % max(2/(1+k/5), 1.0);
end

gamma0 = 1/opts.L0;
gamma = opts.c_gamma * gamma0;% step-size
maxits = opts.maxit+1;
x = x0;
y = x0;
ek = zeros(maxits, 1);
a = opts.a;
S = 1.2;
xi = 0.96;
its = 1;
while(its<maxits)
    x_old = x;
    y_old = y;
    x = proxg(y-gamma*grad_f(y)); % proximal
    y = x + a(its)*(x-x_old);
    %%% gradient criteria
    vk = (y_old(:)-x(:))'*(x(:)-x_old(:));
    if vk >= 0 % restart
        y = x;
    end

    %%%%%%% stop?
    res = norm(x_old-x);
    if mod(its, 1e2)==0
        itsprint(sprintf(' step %08d: residual = %.3e\n', its,res), its);
    end

    ek(its) = res;
    if (res<opts.tol)||(res>1e10)
        break
    end

    %%% safeguard
    if res>S*ek(1)
        gamma = max(gamma0, gamma*xi);
    end
    its = its + 1;

end
fprintf('\n');
fk = fun_f(x)+fun_g(x);
end