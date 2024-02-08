%%
% min ||x||^2_2/2
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

% lambda=1;% lambda;
[m,n]= size(A2);
x0 = rand(n,1);% initial point
maxiter=1e6; % max iterations

%% function definition

% upper level function
fun_f= @(x) x'*x/2;
% lower level function
fun_g = @(x) (A2*x-b2)'*(A2*x-b2)/2;
fun_h = @(x) 0;

% upper level gradient
grad_f = @(x) x;
% lower level gradient
grad_g = @(x) A2'*(A2*x-b2);
% upper level Lipschitz
L_f=1;
% lower level Lipschitz
L_g=eigs(A2'*A2,1);

% find optimal
xx = lsqminnorm(A2'*A2,A2'*b2,1e-8); % least squares least 2norm solution
gstar = fun_g(xx);
fprintf('optimal of lower level = %1.2e\n',gstar);
fprintf('gradient of lower level = %1.2e\n',norm(grad_g(xx)));
fstar = fun_f(xx);
fprintf('optimal of upper level = %f\n',fstar);


%% Initial bounds for Bisec-BiO method
param.L_g = L_g;
param.L_f = L_f;
param.maxiter=5e5;
param.epsilonf = epsilon_f/2;
param.epsilong = epsilon_g/2;
% tolerance of subproblem
param.tol = 1e-8;
tic;
[l,u,x_g,psi_g] = BisecBiO_initial(fun_f,fun_g,fun_h,grad_f,grad_g,x0,param);
time_init0 = toc;


%% Bisec-BiO method
param.maxiter=5e5;
param.epsilonf = epsilon_f/2;
param.epsilong = epsilon_g/2;
param.L0 = L_g;
% tolerance of subproblem
param.tol = 1e-8;
disp('Bisec-BiO method starts');
[f_vec0,g_vec0,time_vec0,x_hat] = BisecBiO(fun_f,fun_g,grad_g,l,u,x_g,psi_g,param);
f_vec0 = [fun_f(x0);fun_f(x_g);f_vec0];g_vec0 = [fun_g(x0);fun_g(x_g);g_vec0];
time_vec0 = time_init0 + time_vec0;
time_vec0 = [0;time_init0;time_vec0];
disp('Bisec-BiO method Achieved!');
fprintf('constraint violation = %1.2e\n',fun_g(x_hat) - gstar);
fprintf('gradient of lower level = %1.2e\n',norm(grad_g(x_hat)));


%% a-IRG Algorithm
param.maxtime = time_vec0(end)+1;
param.maxiter=1e7;
param.L_g = L_g;
disp('a-IRG Algorithm starts')
[f_vec2,g_vec2,time_vec2,xlast] = Alg_Projection(fun_f,grad_f,grad_g,fun_g,param,x0);
f_vec2 = [fun_f(x0);f_vec2];g_vec2 = [fun_g(x0);g_vec2];time_vec2 = [0;time_vec2];
disp('a-IRG Solution Achieved!');


%% BiG-SAM Algorithm
param.eta_g = 1/L_g;
param.eta_f = 1/L_f;
param.gamma = 10;
param.maxtime = time_vec0(end)+1;
disp('BiG-SAM Algorithm starts');
param.maxiter=1e7;
[f_vec3,g_vec3,time_vec3,xlast3] = BigSAM(fun_f,grad_f,grad_g,fun_g,param,x0);
f_vec3 = [fun_f(x0);f_vec3];g_vec3 = [fun_g(x0);g_vec3];time_vec3 = [0;time_vec3];
disp('BiG-SAM Solution Achieved!');


%% MNG
param.maxtime = time_vec0(end)+1;
param.maxiter=1e7;
param.M = L_g;
disp('Mininum norm gradient Algorithm starts');
[f_vec4,g_vec4,time_vec4,xlast4] = MNG(fun_f,grad_f,fun_g,grad_g,param,zeros(n,1));
f_vec4 = [fun_f(x0);f_vec4];g_vec4 = [fun_g(x0);g_vec4];time_vec4 = [0;time_vec4];
disp('MNG Solution Achieved!')


%% DBGD
param.alpha = 1;
param.beta = 1;
param.stepsize = 1e-4;
param.maxiter=1e7;
param.maxtime = time_vec0(end)+1;
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

N_marker = length(g_vec0);
time_idx = linspace(0,param.maxtime,N_marker);
marker_idx0 = zeros(N_marker,1);
marker_idx2 = zeros(N_marker,1);
marker_idx3 = zeros(N_marker,1);
marker_idx4 = zeros(N_marker,1);
marker_idx5 = zeros(N_marker,1);
for j=1:N_marker
    [~,idx] = min(abs(time_vec0-time_idx(j)));
    marker_idx0(j) = idx;
    [~,idx] = min(abs(time_vec2-time_idx(j)));
    marker_idx2(j) = idx;
    [~,idx] = min(abs(time_vec3-time_idx(j)));
    marker_idx3(j) = idx;
    [~,idx] = min(abs(time_vec4-time_idx(j)));
    marker_idx4(j) = idx;
    [~,idx] = min(abs(time_vec5-time_idx(j)));
    marker_idx5(j) = idx;
end
g_gstar = g_vec0-gstar;
n1=find(g_gstar>=epsilon_g);
n2=find(g_gstar<epsilon_g);
absg_gstar = (g_vec0-gstar);
semilogy(time_vec0,(g_vec0-gstar),'p-','DisplayName',' Bisec-BiO','MarkerIndices', marker_idx0);

hold on;
semilogy(time_vec2,(g_vec2-gstar),'s-','DisplayName','a-IRG','MarkerIndices', marker_idx2);
semilogy(time_vec3,(g_vec3-gstar),'^-','DisplayName','BiG-SAM','MarkerIndices', marker_idx3);
semilogy(time_vec4,(g_vec4-gstar),'d-','DisplayName','MNG','MarkerIndices', marker_idx4);
semilogy(time_vec5,(g_vec5-gstar),'>-','DisplayName','DBGD','MarkerIndices', marker_idx5);

ylabel('$g(x_k)-g^*$');
xlabel('time (s)');
set(gca,'FontName','Times New Roman','FontSize',15);
legend('Interpreter','latex','Location','northeast');
absg_gstar_end = sprintf('%.4e',absg_gstar(end));
text(time_vec0(end)+0.07,absg_gstar(end),absg_gstar_end,'FontSize',13,'FontWeight','bold'); % show function value of last iteration
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

semilogy(time_vec0,abs(f_vec0-fstar),'p-','DisplayName',' Bisec-BiO','MarkerIndices', marker_idx0);

hold on;
semilogy(time_vec2,abs(f_vec2-fstar),'s-','DisplayName','a-IRG','MarkerIndices', marker_idx2);
semilogy(time_vec3,abs(f_vec3-fstar),'^-','DisplayName','BiG-SAM','MarkerIndices', marker_idx3);
semilogy(time_vec4,abs(f_vec4-fstar),'d-','DisplayName','MNG','MarkerIndices', marker_idx4);
semilogy(time_vec5,abs(f_vec5-fstar),'>-','DisplayName','DBGD','MarkerIndices', marker_idx5);

ylabel('$|f(x_k)-p^*|$');
xlabel('time (s)');
set(gca,'FontName','Times New Roman','FontSize',15);
legend('Interpreter','latex','Location','northeast');
absf_gstar_end = sprintf('%.4e',f_fstar(end));
text(time_vec0(end)+0.07,absf_gstar(end),absf_gstar_end,'FontSize',13,'FontWeight','bold'); % show function value of last iteration
pbaspect([1 0.8 1])
