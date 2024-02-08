%%
% min ||x||^2_2/2
%s.t. argmin logistic + 1norm ball
clear;clc;close all;
format long
%% load data
seed = 123456;
rng(seed);
addpath('./libsvm-3.3/matlab/'); % use libsvmread
[y, X] = libsvmread('a1a.t');
X = normalize(X,'range');
A = full(X); b = full(y);
% [m,n]= size(A);
A2=A(1:1000,:);A2 = [A2,ones(1000,1)];
b2=b(1:1000);
[m,n]= size(A2);
% solution tolerance
epsilon_f = 1e-5;
epsilon_g = 1e-6;
lambda_g = 10;
x0 = rand(n,1); % initial point
maxiter=1e5; % max iterations

%% function definition

% upper level function
fun_f= @(x) x'*x/2;
% lower level function
fun_g = @(x) (1/m)*ones(1,m)*log(1+exp(-(A2*x).*b2));
fun_h = @(x) 0;

% upper level gradient
grad_f= @(x) x;
% lower level gradient
grad_g = @(x) -(1/m)*A2'*(b2.*(ones(m,1) - 1./(1+exp(-(A2*x).*b2))));
% upper level lipschitz
L_f=1;
% lower level lipschitz
L_g=(0.25/m)*(norm(full(A2),2)^2);

% Finding optimal solution
% lower level optimal: FISTA
opts.L_f = L_f;
opts.L_g = L_g;
opts.L0 = L_g;
opts.tol = 1e-16; % lower-level tolerance
proxh = @(x) ProjectOntoL1Ball(x,lambda_g);
[x_fista, gstar_fista] = Greedy_FISTA_init(fun_g, fun_h, grad_g, proxh, x0, opts);
gstar_fista
norm(x_fista,1)

% total optimal: fmincon
f = @(x) f_logistic(x); g = @(x) g_logistic(x,A2,b2,gstar_fista,epsilon_g,lambda_g);
A=[];b=[];Aeq=[];beq=[];lb=[];ub=[];
options = optimoptions('fmincon','Display','iter','Algorithm','active-set','EnableFeasibilityMode',true, 'HessianApproximation','bfgs', 'SpecifyObjectiveGradient',true, 'MaxFunctionEvaluations',10e6,'MaxIterations',10e6,'ConstraintTolerance',1e-12,'FunctionTolerance',1e-12,'OptimalityTolerance',1e-12,'StepTolerance',1e-12);
[x_fmincon_log,fval,exitflag,output,lambda_fmin,grad] = fmincon(f,x0,A,b,Aeq,beq,lb,ub,g,options);
fstar = fun_f(x_fmincon_log)
fun_g(x_fmincon_log)-gstar_fista
norm(x_fmincon_log,1)
fprintf('\n');


%% Initial bounds for Bisec-BiO method
param.L_g = L_g;
param.L_f = L_f;
param.lam1=lambda_g;
param.maxiter=5e5;
param.epsilonf = epsilon_f/2;
param.epsilong = epsilon_g/2;
param.tol = 1e-8;
tic;
[l,u,x_g,psi_g] = BisecBiO_initial(fun_f,fun_g,fun_h,grad_f,grad_g,x0,param);
time_init0 = toc;


%% Bisec-BiO method
param.lam1=lambda_g;
param.maxiter=5e5;
param.epsilonf = epsilon_f/2;
param.epsilong = epsilon_g/2;
param.L0 = L_g;
disp('Bisec-BiO method starts');
[f_vec0,g_vec0,time_vec0,x_hat] = BisecBiO(fun_f,fun_g,grad_g,l,u,x_g,psi_g,param);
f_vec0 = [fun_f(x0);fun_f(x_g);f_vec0];g_vec0 = [fun_g(x0);fun_g(x_g);g_vec0];
time_vec0 = time_init0 + time_vec0;
time_vec0 = [0;time_init0;time_vec0];
disp('Bisec-BiO method Achieved!');


%% CG for the sub problem
param.epsilong = 1e-2;
param.lam1=lambda_g;
param.maxiter=1e5;
tic;
[x_CG , f_hist] = CG_lowerlevel(fun_g,grad_g,x0,param);
time_init = toc;


%% CG-BiO algorithm
param.epsilonf = epsilon_f;
param.epsilong = epsilon_g;
param.lam=lambda_g;
param.fun_g_x0 = fun_g(x_CG);
param.maxiter=1e5;
param.maxtime = time_vec0(end) + 0.4;
param.inittime = time_init;
disp('CG-BiO starts');
[f_vec1,g_vec1,time_vec1,x] = CG_BiO(fun_f,grad_f,grad_g,fun_g,param,x_CG);
f_vec1 = [fun_f(x0);fun_f(x_CG);f_vec1];g_vec1 = [fun_g(x0);fun_g(x_CG);g_vec1];
disp('CG-BiO Achieved!');
time_vec1 = [0;time_init;time_vec1];


%% a-IRG Algorithm
param.maxtime = time_vec0(end) + 0.4;
param.maxiter=1e7;
disp('a-IRG Algorithm starts')
[f_vec2,g_vec2,time_vec2,xlast] = Alg_Projection(fun_f,grad_f,grad_g,fun_g,param,x0);
f_vec2 = [fun_f(x0);f_vec2];g_vec2 = [fun_g(x0);g_vec2];time_vec2 = [0;time_vec2];
disp('a-IRG Solution Achieved!');


%% BiG-SAM Algorithm
param.eta_g = 1/L_g;
param.eta_f = 1/L_f;
param.gamma = 10;
param.maxtime = time_vec0(end) + 0.4;
disp('BiG-SAM Algorithm starts');
[f_vec3,g_vec3,time_vec3,xlast3] = BigSAM(fun_f,grad_f,grad_g,fun_g,param,x0);
f_vec3 = [fun_f(x0);f_vec3];g_vec3 = [fun_g(x0);g_vec3];time_vec3 = [0;time_vec3];
disp('BiG-SAM Solution Achieved!');


%% MNG
param.maxtime = time_vec0(end) + 0.4;
param.maxiter=1e5;
param.M = L_g;
disp('Mininum norm gradient Algorithm starts');
[f_vec4,g_vec4,time_vec4,xlast4] = MNG(fun_f,grad_f,fun_g,grad_g,param,zeros(n,1));
f_vec4 = [fun_f(x0);f_vec4];g_vec4 = [fun_g(x0);g_vec4];time_vec4 = [0;time_vec4];
disp('MNG Solution Achieved!')


%% DBGD
param.alpha = 1;
param.beta = 1;
param.stepsize = 1/L_g;
param.maxiter=1e5;
param.maxtime = time_vec0(end) + 0.4;
disp('DBGD Algorithm starts');
[f_vec5,g_vec5,time_vec5,xlast5] = DBGD(fun_f,grad_f,grad_g,fun_g,param,x0);
f_vec5 = [fun_f(x0);f_vec5];g_vec5 = [fun_g(x0);g_vec5];time_vec5 = [0;time_vec5];
disp('DBGD Solution Achieved!');


%% Figures
%% lower-level gap
figure (1);
set(0,'defaulttextinterpreter','latex')
set(gcf,'DefaultLineLinewidth',3)
set(gcf,'DefaultLineMarkerSize',4);
set(gcf,'Position',[331,167,591,586])

N_marker = length(f_vec0);
time_idx = linspace(0,param.maxtime,N_marker);
marker_idx0 = zeros(N_marker,1);
marker_idx1 = zeros(N_marker,1);
marker_idx2 = zeros(N_marker,1);
marker_idx3 = zeros(N_marker,1);
marker_idx4 = zeros(N_marker,1);
marker_idx5 = zeros(N_marker,1);
marker_idx_GD = zeros(N_marker,1);
for j=1:N_marker
    [~,idx] = min(abs(time_vec0-time_idx(j)));
    marker_idx0(j) = idx;
    [~,idx] = min(abs(time_vec1-time_idx(j)));
    marker_idx1(j) = idx;
    [~,idx] = min(abs(time_vec2-time_idx(j)));
    marker_idx2(j) = idx;
    [~,idx] = min(abs(time_vec3-time_idx(j)));
    marker_idx3(j) = idx;
    [~,idx] = min(abs(time_vec4-time_idx(j)));
    marker_idx4(j) = idx;
    [~,idx] = min(abs(time_vec5-time_idx(j)));
    marker_idx5(j) = idx;
end
g_gstar = g_vec0-gstar_fista;
n1=find(g_gstar>=epsilon_g);
n2=find(g_gstar<epsilon_g);
absg_gstar = (g_vec0-gstar_fista);
semilogy(time_vec0,(g_vec0-gstar_fista),'p-','DisplayName','Bisec-BiO','MarkerIndices', marker_idx0);

hold on;
semilogy(time_vec1,(g_vec1-gstar_fista),'o-','DisplayName','CG-BiO','MarkerIndices', marker_idx1);
semilogy(time_vec2,(g_vec2-gstar_fista),'s-','DisplayName','a-IRG','MarkerIndices', marker_idx2);
semilogy(time_vec3,(g_vec3-gstar_fista),'^-','DisplayName','BiG-SAM','MarkerIndices', marker_idx3);
semilogy(time_vec4,(g_vec4-gstar_fista),'d-','DisplayName','MNG','MarkerIndices', marker_idx4);
semilogy(time_vec5,(g_vec5-gstar_fista),'>-','DisplayName','DBGD','MarkerIndices', marker_idx5);

ylabel('$g(x_k)-g^*$');
xlabel('time (s)');
set(gca,'FontName','Times New Roman','FontSize',15);
legend('Interpreter','latex','Location','northeast');
absg_gstar_end = sprintf('%.4e',absg_gstar(end));
text(time_vec0(end)+0.03,absg_gstar(end),absg_gstar_end,'FontSize',13,'FontWeight','bold'); % show function value of last iteration
pbaspect([1 0.8 1])


%% upper-level gap
figure (2);
set(0,'defaulttextinterpreter','latex')
set(gcf,'DefaultLineLinewidth',3)
set(gcf,'DefaultLineMarkerSize',4);
set(gcf,'Position',[331,167,591,586])
f_fstar = f_vec0-fstar;
n1=find(f_fstar>=epsilon_f);
n2=find(f_fstar<epsilon_f);
absf_gstar = abs(f_vec0-fstar);
f_fstar = f_vec0-fstar;

semilogy(time_vec0,abs(f_vec0-fstar),'p-','DisplayName','Bisec-BiO','MarkerIndices', marker_idx0);

hold on;
semilogy(time_vec1,abs(f_vec1-fstar),'o-','DisplayName','CG-BiO','MarkerIndices', marker_idx1);
semilogy(time_vec2,abs(f_vec2-fstar),'s-','DisplayName','a-IRG','MarkerIndices', marker_idx2);
semilogy(time_vec3,abs(f_vec3-fstar),'^-','DisplayName','BiG-SAM','MarkerIndices', marker_idx3);
semilogy(time_vec4,abs(f_vec4-fstar),'d-','DisplayName','MNG','MarkerIndices', marker_idx4);
semilogy(time_vec5,abs(f_vec5-fstar),'>-','DisplayName','DBGD','MarkerIndices', marker_idx5);

ylabel('$|f(x_k)-p^*|$');
xlabel('time (s)');
set(gca,'FontName','Times New Roman','FontSize',15);
legend('Interpreter','latex','Location','northeast');
absf_gstar_end = sprintf('%.4e',f_fstar(end));
text(time_vec0(end)+0.03,absf_gstar(end),absf_gstar_end,'FontSize',13,'FontWeight','bold'); % show function value of last iteration
pbaspect([1 0.8 1])
