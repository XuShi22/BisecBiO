%%
% min ||x||_1 + alpha*||x||^2_2/2
%s.t. argmin ||Ax-b||_2^2
clear;clc;close all;
format long
%% load data
seed = 123456;
rng(seed);
X = load('YearPredictionMSD.txt');
[mm,~] = size(X);
num = 1000; % train number
co = randperm(mm,num); XX = X(co',:); % randomly select 1000 rows
bb = XX(:,1);AA = XX(:,2:end); %bb: label，AA: feature matrix
AA = normalize(AA,'range',[-1,1]); %scaling feature matrix to [-1,1]
AA = [AA,ones(num,1)]; % add an interceptor term
A = full(AA);b = full(bb);
b = normalize(b,'range');
[m,n]= size(A);

% add co-linear attributes to A
A_tilde = zeros(m,n-1);
for i = 1:n-1
    idx = randperm(n-1,10);
    B = A(:,idx); % generate a matrix contain 10 column of A
    v = unifrnd(-1,1,[10,1]);
    A_tilde(:,i) = B*v;
end
A = [A,A_tilde];
A2=A;
b2=b;
% solution tolerance
epsilon_f = 1e-5;
epsilon_g = 1e-6;
[m,n]= size(A2);

x0 = rand(n,1);% initial point
maxiter=1e6; % max iterations

%% function definition

% upper level function
alpha = 0.02; fun_f= @(x) sum(abs(x)) + alpha*(x'*x)/2;
% lower level function
fun_g = @(x) (A2*x-b2)'*(A2*x-b2)/2;
fun_h = @(x) 0;

%upper gradient
grad_f = @(x) 0; % donot need
%lower gradient
grad_g = @(x) A2'*(A2*x-b2);

%upper Lipschitz
L_f=sqrt(n) + alpha;
%lower Lipschitz
L_g=eigs(A2'*A2,1);
A2 = full(A2);b2=full(b2);

% Find optimal
% lower optimal: lsqminnorm
xx = lsqminnorm(A2'*A2,A2'*b2,1e-8);
gstar = fun_g(xx)
norm(grad_g(xx))

% total optimal: fmincon
f = @(x) f_linear(x,alpha); g = @(x) g_linear(x,A2,b2,gstar);
A=[];b=[];Aeq=[];beq=[];lb=[];ub=[];
options = optimoptions('fmincon','Display','iter','Algorithm','interior-point','EnableFeasibilityMode',true,'HessianApproximation','bfgs','MaxFunctionEvaluations',10e6,'MaxIterations',5e4,'ConstraintTolerance',1e-15,'OptimalityTolerance',1e-15,'StepTolerance',1e-15,'FunctionTolerance',1e-15);
[x_fmincon,fval,exitflag,output,lambda_fmin,grad] = fmincon(f,x0,A,b,Aeq,beq,lb,ub,g,options);
fstar_fmincon = fun_f(x_fmincon)
fun_g(x_fmincon)-gstar
x_fmincon = full(x_fmincon);
norm(grad_g(x_fmincon))


%% Initial bounds for Bisec-BiO method
param.L_g = L_g;
param.L_f = L_f;
param.maxiter=1e6;
param.epsilonf = epsilon_f/2;
param.epsilong = epsilon_g/2;
param.alpha = alpha;
param.tol = 1e-8;
tic;
[l,u,x_g,psi_g] = BisecBiO_initial(fun_f,fun_g,fun_h,grad_f,grad_g,x0,param);
time_init0 = toc;


%% Bisec-BiO method
param.maxiter=1e6;
param.epsilonf = epsilon_f/2;
param.epsilong = epsilon_g/2;
param.L0 = L_g;
disp('Bisec-BiO method starts');
[f_vec0,g_vec0,time_vec0,x_hat] = BisecBiO(fun_f,fun_g,grad_g,l,u,x_g,psi_g,param);
f_vec0 = [fun_f(x0);fun_f(x_g);f_vec0];g_vec0 = [fun_g(x0);fun_g(x_g);g_vec0];
time_vec0 = time_init0 + time_vec0;
time_vec0 = [0;time_init0;time_vec0];
disp('Bisec-BiO method Achieved!');

fprintf('constraint violation = %1.2e\n',fun_g(x_hat) - gstar);
fprintf('gradient of lower level = %1.2e\n',norm(grad_g(x_hat)));


%% a-IRG Algorithm
param.maxtime = time_vec0(end)+3;
param.maxiter=1e7;
subgrad_f = @(x) subgrad_f_alg(x,alpha);
disp('a-IRG Algorithm starts')
[f_vec2,g_vec2,time_vec2,xlast] = Alg_Projection(fun_f,subgrad_f,grad_g,fun_g,param,x0);
f_vec2 = [fun_f(x0);f_vec2];g_vec2 = [fun_g(x0);g_vec2];time_vec2 = [0;time_vec2];
disp('a-IRG Solution Achieved!');


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
marker_idx2 = zeros(N_marker,1);
for j=1:N_marker
    [~,idx] = min(abs(time_vec0-time_idx(j)));
    marker_idx0(j) = idx;
    [~,idx] = min(abs(time_vec2-time_idx(j)));
    marker_idx2(j) = idx;
end
g_gstar = g_vec0-gstar;
n1=find(g_gstar>=epsilon_g);
n2=find(g_gstar<epsilon_g);
absg_gstar = (g_vec0-gstar);
semilogy(time_vec0,(g_vec0-gstar),'p-','DisplayName','Bisec-BiO','MarkerIndices', marker_idx0);

hold on;
semilogy(time_vec2,(g_vec2-gstar),'s-','DisplayName','a-IRG','MarkerIndices', marker_idx2);

ylabel('$g(x_k)-g^*$');
xlabel('time (s)');
set(gca,'FontName','Times New Roman','FontSize',15);
legend('Interpreter','latex','Location','northeast');
absg_gstar_end = sprintf('%.4e',absg_gstar(end));
text(time_vec0(end)+0.2,absg_gstar(end),absg_gstar_end,'FontSize',13,'FontWeight','bold'); % show function value of last iteration
pbaspect([1 0.8 1])


%% upper-level gap
figure (2);
set(0,'defaulttextinterpreter','latex')
set(gcf,'DefaultLineLinewidth',3)
set(gcf,'DefaultLineMarkerSize',4);
set(gcf,'Position',[331,167,591,586])
f_fstar = f_vec0-fstar_fmincon;
n1=find(f_fstar>=epsilon_f);
n2=find(f_fstar<epsilon_f);
absf_gstar = abs(f_vec0-fstar_fmincon);
f_fstar = f_vec0-fstar_fmincon;

semilogy(time_vec0,abs(f_vec0-fstar_fmincon),'p-','DisplayName','Bisec-BiO','MarkerIndices', marker_idx0);

hold on;
semilogy(time_vec2,abs(f_vec2-fstar_fmincon),'s-','DisplayName','a-IRG','MarkerIndices', marker_idx2);

ylabel('$|f(x_k)-p^*|$');
xlabel('time (s)');
set(gca,'FontName','Times New Roman','FontSize',15);
legend('Interpreter','latex','Location','northeast');
absf_gstar_end = sprintf('%.4e',f_fstar(end));
text(time_vec0(end)+0.2,absf_gstar(end),absf_gstar_end,'FontSize',13,'FontWeight','bold'); % show function value of last iteration
pbaspect([1 0.8 1])
