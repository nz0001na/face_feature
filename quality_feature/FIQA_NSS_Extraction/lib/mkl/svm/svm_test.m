% Support Vector Machine (SVM)

% Summary
%   tests SVM on test data with given model

% Input(s)
%   tes: test data
%   mod: SVM model

% Output(s)
%   out: classification outputs

% Mehmet Gonen (gonen@boun.edu.tr)
% Bogazici University
% Department of Computer Engineering

function out = svm_test(tes, mod)
    tes.X = normalize_data(tes.X, mod.nor.dat);
    K = kernel(tes, mod.sup, mod.par.ker, mod.par.nor.ker);
    out.dis = K * (mod.sup.alp .* mod.sup.y) + mod.b;
end