%% IMPORTANT:
% MATLAB Needs to be run with following command to make this work:
% env LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 ./MATLAB/R2015b/bin/matlab -desktop

% addpath('../lib/heuristic_features/face-release1.0-basic');

% load and visualize model
% Pre-trained model with 146 parts. Works best for faces larger than 80*80
% load face_p146_small.mat

function angl = pose_estimation(image_path, model)

im = imread(image_path);

% 5 levels for each octave
model.interval = 5;
% set up the threshold
model.thresh = min(-0.65, model.thresh);

% define the mapping from view-specific mixture id to viewpoint
if length(model.components)==13 
    posemap = 90:-15:-90;
elseif length(model.components)==18
    posemap = [90:-15:15 0 0 0 0 0 0 -15:-15:-90];
else
    error('Can not recognize this model');
end

try
    bs = detect(im, model, model.thresh);
    bs = clipboxes(im, bs);
    bs = nms_face(bs,0.3);
    angl = posemap(bs(1).c);
catch 
    angl= intmax;
end