% Rule-Based Multiple Kernel Support Vector Machine (RBMKSVM)

% Summary
%   creates a default parameter set for RBMKSVM

% Output(s)
%   par: constructed parameter set

% Mehmet Gonen (gonen@boun.edu.tr)
% Bogazici University
% Department of Computer Engineering

function par = rbmksvm_parameter()
    par.C = 1; % regularization parameter
    par.eps = 1e-3; % threshold parameter
    par.ker = {'l', 'p2'}; % kernel functions [l: linear, p:polynomial, g:gaussian]
    par.nor.dat = {'true', 'true'}; % if true, apply z-normalization to data
    par.nor.ker = {'true', 'true'}; % if true, make kernel unit diagonal
    par.opt = 'smo'; % optimizer [libsvm, mosek, quadprog, smo]
    par.rul = 'mean'; % combination rule [mean, product]
    par.tau = 1e-3; % tau parameter for SMO algorithm
end