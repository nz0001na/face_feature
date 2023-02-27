% Support Vector Machine (SVM)

% Summary
%   creates a default parameter set for SVM

% Output(s)
%   par: constructed parameter set

% Mehmet Gonen (gonen@boun.edu.tr)
% Bogazici University
% Department of Computer Engineering

function par = svm_parameter()
    par.C = 1; % regularization parameter
    par.eps = 1e-3; % threshold parameter
    par.ker = 'l'; % kernel function [l: linear, p:polynomial, g:gaussian]
    par.nor.dat = 'true'; % if true, apply z-normalization to data
    par.nor.ker = 'true'; % if true, make kernel unit diagonal
    par.opt = 'smo'; % optimizer [libsvm, mosek, quadprog, smo]
    par.tau = 1e-3; % tau parameter for SMO algorithm
end