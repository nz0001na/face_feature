clear;clc;
addpath('C:\Naved\face_quality_research\face-release1.0-basic');

% load and visualize model
% Pre-trained model with 146 parts. Works best for faces larger than 80*80
load face_p146_small.mat

% % Pre-trained model with 99 parts. Works best for faces larger than 150*150
% load face_p99.mat

% % Pre-trained model with 1050 parts. Give best performance on localization, but very slow
% load multipie_independent.mat

% disp('Model visualization');
% visualizemodel(model,1:13);
% disp('press any key to continue');
% pause;


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

src = 'C:\Naved\Face_Databases\CALTECH_10k_Webfaces\Caltech_WebFaces\';
dst = 'C:\Naved\Face_Quality_Database\real_world\caltech_10k_webfaces\';

% d = dir(src);
% isub = [d(:).isdir]; %# returns logical vector
% nameFolds = {d(isub).name}';
% nameFolds(ismember(nameFolds,{'.','..'})) = [];
% 
% for j = 1:length(nameFolds)
%     fprintf('testing: %d/%d\n', j, length(nameFolds));
%     ims = dir([src nameFolds{j} '\*.jpg']);
%     count = 0;
%     for i = 1:length(ims),
%         im = imread([src nameFolds{j} '\' ims(i).name]);
%         try
%             bs = detect(im, model, model.thresh);
%             if ~isempty(bs)
%                 bs = clipboxes(im, bs);
%                 bs = nms_face(bs,0.3);
%                 angl = posemap(bs(1).c);
%                 if angl == 0 %frontal
%                     copyfile([src nameFolds{j} '\' ims(i).name],[dst nameFolds{j} '_' ims(i).name]);
%                     count = count + 1;
%                     display(['frontal face detected in ' ims(i).name ', count=' num2str(count)]);
%                 end
%             end
%         catch exception
%             continue;
%         end
%     end
% end
% disp('done!');

%%%%%%%%%%%%%%%%%%%

ims = dir([src '*.jpg']);
count = 0;
for i = 1:length(ims),
    fprintf('testing: %d/%d\n', i, length(ims));
    im = imread([src ims(i).name]);
    try
        bs = detect(im, model, model.thresh);
        if ~isempty(bs)
            bs = clipboxes(im, bs);
            bs = nms_face(bs,0.3);
            angl = posemap(bs(1).c);
            if angl == 0 %frontal
                copyfile([src ims(i).name],[dst ims(i).name]);
                count = count + 1;
                display(['frontal face detected in ' ims(i).name ', count=' num2str(count)]);
            end
        end
    catch exception
        continue;
    end
end
disp('done!');