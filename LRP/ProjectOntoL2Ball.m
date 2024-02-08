function w = ProjectOntoL2Ball(v, b)
% PROJECTONTOL2BALL Projects point onto L2 ball of specified radius.
%
% w = ProjectOntoL1Ball(v, b) returns the vector w which is the solution
% to the following constrained minimization problem:

%    min   ||w - v||_2
%    s.t.  ||w||_2 <= b.

if (b < 0)
    error('Radius of L2 ball is negative: %2.3f\n', b);
end
if (norm(v, 2) < b)
    w = v;
    return;
end
w=b*v/norm(v, 2);
end