function prepare()
	pat = pwd;
    addpath(sprintf('%s', pat));

	addpath(sprintf('%s/common', pat));
	addpath(sprintf('%s/common/data', pat));
	addpath(sprintf('%s/common/drawing', pat));
	addpath(sprintf('%s/common/gradient', pat));
	addpath(sprintf('%s/common/gui', pat));
	addpath(sprintf('%s/common/kernel', pat));
	addpath(sprintf('%s/common/locality', pat));
	addpath(sprintf('%s/common/multikernel', pat));		
	addpath(sprintf('%s/common/solvers', pat));

    addpath(sprintf('%s/abmksvm', pat));
    addpath(sprintf('%s/cabmksvm', pat));
    addpath(sprintf('%s/glmksvm', pat));
    addpath(sprintf('%s/gmksvm', pat));
    addpath(sprintf('%s/lmksvm', pat));
    addpath(sprintf('%s/mksvm', pat));
    addpath(sprintf('%s/nlmksvm', pat));
    addpath(sprintf('%s/rbmksvm', pat));
    addpath(sprintf('%s/smksvm', pat));
    addpath(sprintf('%s/svm', pat));
end