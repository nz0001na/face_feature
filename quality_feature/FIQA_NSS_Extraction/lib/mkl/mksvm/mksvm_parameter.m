% Multiple Kernel Support Vector Machine (MKSVM)

% Summary
%   creates a default parameter set for MKSVM

% Output(s)
%   par: constructed parameter set

% Mehmet Gonen (gonen@boun.edu.tr)
% Bogazici University
% Department of Computer Engineering

function par = mksvm_parameter()
    par.C = 1; % regularization parameter
    par.eps = 1e-3; % threshold parameter
    par.ker = {'l', 'p2'}; % kernel functions [l: linear, p:polynomial, g:gaussian]
    par.nor.dat = {'true', 'true'}; % if true, apply z-normalization to data
    par.nor.ker = {'true', 'true'}; % if true, make kernel unit diagonal
    par.opt = 'mosek'; % optimizer [mosek]
end