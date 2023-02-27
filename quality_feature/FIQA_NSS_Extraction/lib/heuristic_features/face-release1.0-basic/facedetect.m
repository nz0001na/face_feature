clear;clc;
% addpath('C:\Naved\face_quality_research\face-release1.0-basic');

% load and visualize model
% Pre-trained model with 146 parts. Works best for faces larger than 80*80
% load face_p146_small.mat

% % Pre-trained model with 99 parts. Works best for faces larger than 150*150
load face_p99.mat

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

%%%%%%%%%%%%%%%%%%%

im = imread('G:\ROOTDIR\FQDB\Originals\pics2d_iranian\90\if1a05.JPG');
if size(im,3) == 1 % if grayscale, convert to rgb
    im = repmat(im,[1 1 3]);
end
%     im = imresize(im, [250 250]); 
bs = detect(im, model, model.thresh);
if ~isempty(bs)
    bs = clipboxes(im, bs);
    bs = nms_face(bs,0.3);
    angl = posemap(bs(1).c);
    % show highest scoring one
    figure;

    %% showboxes(im, boxes)
    % Draw boxes on top of image.
    imagesc(im);
    hold on;
    axis image;
    axis off;
    boxes = bs(1);
    avgpoints = zeros(length(boxes.xy),2);
    for b = boxes,
        partsize = b.xy(1,3)-b.xy(1,1)+1;
        tx = (min(b.xy(:,1)) + max(b.xy(:,3)))/2;
        ty = min(b.xy(:,2)) - partsize/2;
        text(tx,ty, num2str(posemap(b.c)),'fontsize',18,'color','c');
        for i = size(b.xy,1):-1:1;
            x1 = b.xy(i,1);
            y1 = b.xy(i,2);
            x2 = b.xy(i,3);
            y2 = b.xy(i,4);

            plot((x1+x2)/2,(y1+y2)/2,'r.','markersize',15);
            avgpoints(i,1)= (x1+x2)/2;
            avgpoints(i,2)= (y1+y2)/2;
        end
    end
    title('Highest scoring detection');
    drawnow;
     
    if posemap(b.c) > 0
        left = (min(avgpoints(:,1)) + max(avgpoints(:,1))) / 2; % for right profile faces
        top = min(avgpoints(:,2));
        width = max(avgpoints(:,1))-left;
        height = max(avgpoints(:,2))-top;
    elseif posemap(b.c) < 0
        left = min(avgpoints(:,1)); % for left profile faces
        top = min(avgpoints(:,2));
        width = max(avgpoints(:,1)) / 2;
        height = max(avgpoints(:,2))-top;    
    else
        left = min(avgpoints(:,1)); % frontal/semi-frontal faces  
        top = min(avgpoints(:,2));
        width = max(avgpoints(:,1))-left;
        height = max(avgpoints(:,2))-top; 
    end
    I2 = imcrop(im, [left top width height]);
    figure;imshow(I2);
else
    display('failed to detect faces.');
end

