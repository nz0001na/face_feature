%  
function hf = heuristic_features(I, file_path)

    % read image, its the detected face region, aligned, converted to
    % grayscale and resized to 100x60.
    crgr = double(I);

    %% brightness: since the region of the face is usually a small region we can
    % consider the average value of the illumination component of all of the
    % pixels in this region as the brightness of that region. (Nasrollahi and
    % Moeslund BIM 2008, Haque et al. AVSS 2013)
%     brightness = sum(sum(crgr))/(size(crgr,1)*size(crgr,2));
    brightness = mean2(crgr);

    %% contrast: image contrast is the difference in color intensities that
    % makes a face distinguishable. The face image contrast can be measured
    % using the following equation: C = sqrt(sum(sum((I-mean(I))^2)) / M*N )
    % (Abaza et al. 2014)
    contrast = sqrt(sum(sum((crgr-mean2(crgr)).^2)) / (size(crgr,1)*size(crgr,2)));
    
    %% focus: focus refers to the degree of blurring of face images. 
    [Gx, Gy] = imgradientxy(crgr);
    [Gmag, Gdir] = imgradient(Gx, Gy);
    focus = mean2(Gmag); % edge density    
    
    %% Illumination: Spectral energy (Nill and Bouzas 1992)
    illum_crgr = imresize(crgr, [256 256]);
    blockSize = [size(illum_crgr,1)/16 size(illum_crgr,2)/16];
    fun = @(block_struct) abs(fft2(block_struct.data));
    B = blockproc(illum_crgr,blockSize,fun);  
    illum = sum(sum(B));     
    
    %% Illumination symmetry
    leftHalf = crgr(:,1:round(size(crgr,2)/2));
    rightHalf = crgr(:,round(size(crgr,2)/2)+1:end);
    illum_sym = abs(mean2(leftHalf)-mean2(rightHalf));     
    
    %% sharpness: if I is the face region, and LowPass(I) is the result of
    % appling a low-pass filter to it, then the average value of the pixels of
    % the equation E = |I-Lowpass(I)| is the sharpness of the face (Nasrollahi and
    % Moeslund BIM 2008, Haque et al. AVSS 2013)
    h = fspecial('gaussian', [3 3], 0.5);
    filtered_crgr = imfilter(crgr, h);
    sharpness = sum(sum(crgr - filtered_crgr)) / (size(crgr,1)*size(crgr,2));
    
    %% Compression
    compression = jpeg_quality_score(crgr);
    
    %% Pose estimation
%     pose estimation method proposed by Zhu, Xiangxin, and Deva Ramanan in "Face detection, pose estimation, 
%     and landmark localization in the wild." Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.    
%     pose_angle = pose_estimation(file_path, model);
    % get pose angle from pts files
    [euler_x, euler_y, euler_z] = get_pose_angles(file_path);
    %% eye openness
    LeftEyeDetect = vision.CascadeObjectDetector('LeftEye','MergeThreshold',16);
    leBB=step(LeftEyeDetect,I);
    RightEyeDetect = vision.CascadeObjectDetector('RightEye','MergeThreshold',16);
    reBB=step(RightEyeDetect,I);
    % figure,imshow(I);
    % for i = 1:size(leBB,1)
    %  rectangle('Position',leBB(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','b');
    % end
    % for i = 1:size(reBB,1)
    %  rectangle('Position',reBB(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','g');
    % end
    % title('Eyes Detection');

    % to handle multiple detection:
    % select left most bbox for left eye, select right most bbox for right eye
    if ~isempty(leBB) && ~isempty(reBB)
        [min_x, min_idx] = min(leBB(:,1));   
        [max_x, max_idx] = max(reBB(:,1));
        leBB = leBB(min_idx,:);
        reBB = reBB(max_idx,:);
    % openness of eyes: (Nasrollahi et al. 2011) used the aspect ratio of the
    % eye's height to its width. The bigger the ratio is, the more open the eye
    % is.
        width = leBB(3);
        height = leBB(4);
        left_eye_ratio = width / height;

        width = reBB(3);
        height = reBB(4);
        right_eye_ratio = width / height;
    else
        left_eye_ratio = 0; % to handle no detection (eg. partially occluded faces)
        right_eye_ratio = 0; % to handle no detection (eg. partially occluded faces)
    end        
    %% mouth closeness
    %To detect Mouth
    MouthDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',16);
    mouthBB = step(MouthDetect,I); % BB=[x y width height], (x,y)=upper-left corner
    % figure,
    % imshow(I); hold on
    % for i = 1:size(mouthBB,1)
    %  rectangle('Position',mouthBB(i,:),'LineWidth',4,'LineStyle','-','EdgeColor','r');
    % end
    % title('Mouth Detection');
    % hold off;

    % to handle multiple detection:
    %select the box with the maximum area
    if ~isempty(mouthBB)
        [area, idx] = max(mouthBB(:,3) .* mouthBB(:,4));
        mouthBB = mouthBB(idx,:);
    % mouth closeness: detect the mouth and calculate the ratio between height
    % and width of mouth region. The closer is the mouth, the smaller is this
    % ratio. (Nasrollahi et al. 2011, Anantharajah et al. 2012)
        width = mouthBB(3);
        height = mouthBB(4);
        mouth_ratio = width / height;
    else
        mouth_ratio = 0; % to handle no detection (eg. partially occluded faces)
    end    
    %% Face symmmetry: 
    % (Sang et al. 2009 ICB) also can be used to evaluate lighting symmetry    
    leftHalf = crgr(:,1:round(size(crgr,2)/2));
    lh_gaborf = gaborResponse(leftHalf, 'imag');
    rightHalf = crgr(:,round(size(crgr,2)/2)+1:end);
    rh_gaborf = gaborResponse(rightHalf, 'imag');
    mirror_rh_gaborf = fliplr(rh_gaborf); 
    face_sym = sum(lh_gaborf+mirror_rh_gaborf); % lower the value, better the symmetry
    
    %% concatenate all heuristic values
    hf = [brightness contrast focus illum illum_sym sharpness compression ...
        euler_x euler_y euler_z left_eye_ratio right_eye_ratio mouth_ratio face_sym]; % 1x12 
%     hf(isnan(hf)) = 0; % replace NaN with 0.
%     hf = double(hf);
end