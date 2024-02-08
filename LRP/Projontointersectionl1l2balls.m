function x = Projontointersectionl1l2balls(v, alpha, beta)
% Projection on the insection of L1ball and L2 ball in "Projections onto the Intersection of a One-Norm Ball or
% Sphere and a Two-Norm Ball or Sphere",
% H. Liu, H. Wang and M. Song, J Optim Theory Appl. 2020

% This function solves the optimization problem:
% min 1/2 ||x- v||_2^2 s.t. ||x||_1 <= alpha, ||x||_2 <= beta
% Input:
% v: vector
% alpha: L1 norm constraint
% beta: L2 norm constraint
% Output:
% x: optimal solution

d=length(v);
if alpha<=beta
    x=ProjectOntoL1Ball(v,alpha);
elseif alpha>=sqrt(d)*beta
    x=ProjectOntoL2Ball(v,beta);
else
    if norm(v,1)<= alpha && norm(v,2)<= beta
        x=v;
    else
        x1=ProjectOntoL1Ball(v,alpha);
        x2=ProjectOntoL2Ball(v,beta);
        [~,  xp, ~, ~, ~]=S1S2p_QASB(v/beta, d, alpha/beta);
        x3=beta*xp;
        if norm(x1,2)<=beta
            x=x1;
        elseif norm(x2,1)<=alpha
            x=x2;
        else
            x=x3;
        end
    end
end
end