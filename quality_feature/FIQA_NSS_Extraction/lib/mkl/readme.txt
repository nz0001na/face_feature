---------------------------------------------------------------------------

If you use the source code, please cite the following paper:

@article{gonen11jmlr,
    Author = {G\"{o}nen, Mehmet and Alpayd{\i}n, Ethem},
    Journal = {Journal of Machine Learning Research},
    Number = {Jul},
    Pages = {2211--2268},
    Title = {Multiple Kernel Learning Algorithms},
    Volume = {12},
    Year = {2011}}

---------------------------------------------------------------------------

You should first run the script "prepare.m" in order to add the necessary
folders to MATLAB's path.

---------------------------------------------------------------------------

Each classifier is implemented by three files: parameters (*_parameter.m),
train function (*_train.m), and test function (*_test.m).

---------------------------------------------------------------------------

There are two demo scripts showing how to train and to test the classifiers.
rbmksvm_demo.m => trains RBMKL (mean) on a toy data set
lmksvm_demo.m  => trains LMKL (softmax) on a toy data set 

---------------------------------------------------------------------------

Default optimizer is set to an SMO solver written in MATLAB. Optimizer can
be changed to LIBSVM or MOSEK after installing these packages. Please see
the demo scripts to learn how to change the default optimizer.

LIBSVM => http://www.csie.ntu.edu.tw/~cjlin/libsvm/
MOSEK => http://www.mosek.com/

---------------------------------------------------------------------------

The following list matches the algorithms used in the paper and the code 
files provided.

SVM (best)       => svm/svm_*.m
SVM (all)        => svm/svm_*.m
RBMKL (mean)     => rbmksvm/rbmksvm_*.m         rul = 'mean'
RBMKL (product)  => rbmksvm/rbmksvm_*.m         rul = 'product'
ABMKL (conic)    => abmksvm/abmksvm_*.m         com = 'conic'
ABMKL (convex)   => abmksvm/abmksvm_*.m         com = 'convex'
ABMKL (ratio)    => abmksvm/abmksvm_*.m         com = 'ratio'
CABMKL (linear)  => cabmksvm/cabmksvm_*.m       com = 'linear'
CABMKL (conic)   => cabmksvm/cabmksvm_*.m       com = 'convex'
MKL(best)        => mksvm/mksvm_*.m        
SimpleMKL (best) => mksvm/mksvm_*.m
GMKL             => gmksvm/gmksvm_*.m
GLMKL (p = 1)    => glmksvm/glmksvm_*.m         p = 1
GLMKL (p = 2)    => glmksvm/glmksvm_*.m         p = 2
NLMKL (p = 1)    => nlmksvm/nlmksvm_*.m         p = 1
NLMKL (p = 2)    => nlmksvm/nlmksvm_*.m         p = 2
LMKL (softmax)   => lmksvm/lmksvm_*.m           gat.type = 'linear_softmax'
LMKL (sigmoid)   => lmksvm/lmksvm_*.m           gat.type = 'linear_sigmoid'

---------------------------------------------------------------------------