% Support Vector Machine (SVM)

% Reference
%   vapnik98book
%   Vladimir N. Vapnik
%   Statistical Learning Theory
%   John Wiley & Sons, 1998

% Summary
%   trains SVM on training data with given parameters

% Input(s)
%   tra: training data
%   par: parameters

% Output(s)
%   mod: SVM model

% Mehmet Gonen (gonen@boun.edu.tr)
% Bogazici University
% Department of Computer Engineering

function mod = svm_train(tra, par)
    mod.nor.dat = mean_and_std(tra.X, par.nor.dat);
    tra.X = normalize_data(tra.X, mod.nor.dat);
    yyK = (tra.y * tra.y') .* kernel(tra, tra, par.ker, par.nor.ker);
    N = size(tra.X, 1);
    alp = zeros(N, 1);
    alp = solve_svm(tra, par, yyK, alp);
    sup = find(alp ~= 0);
    act = find(alp ~= 0 & alp < par.C);
    mod.sup.ind = tra.ind(sup);
    mod.sup.X = tra.X(sup, :);
    mod.sup.y = tra.y(sup);
    mod.sup.alp = alp(sup);
    if isempty(act) == 0
        mod.b = mean(tra.y(act) .* (1 - yyK(act, sup) * alp(sup)));
    else
        mod.b = 0;
    end
    mod.par = par;
end