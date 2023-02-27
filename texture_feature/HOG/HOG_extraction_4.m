% this code is to extract HoG feature of images

% Example: Running the code
ro = 'E:/1_research/1_MAD/1_Data/'
src_path = [ro '4_StyleGAN_Morphs_data/6_cropped_resize/']
dst_path = [ro '8_texture_feature/4_StyleGAN/HOG/']
if ~exist(dst_path, 'dir')
    mkdir(dst_path)
end

d       = dir(fullfile(src_path));
dirlist = d([d.isdir]);
dirlist = dirlist(~ismember({dirlist.name}, {'.','..'}));
% a  =size(dirlist,1)
for i=1:size(dirlist,1)
    folder = dirlist(i).name
    if strcmp(folder,'2_StyleGANMorphs_jpg')~=0
        continue
    end
    if strcmp(folder,'2_StyleGANMorphs_png')~=0
        continue
    end

    folder_path = [src_path folder]

    if ~exist([dst_path folder], 'dir')
        mkdir([dst_path folder])
    end
    % ?????? .jpg
    Images = dir([folder_path, '\*.jpg']);
    len = size(Images,1)
    if len == 0
        continue
    end
    for j=1:len
        img_name = Images(j).name
        a = strsplit(img_name,'.')
        new_name = [a{1} '.mat']

        if exist([dst_path folder '/' new_name ], 'file')
            continue
        end
        img = [src_path folder '/' img_name]
        I= imread(img);  
        feature = hog_feature_vector(I);
        save([dst_path folder '\' new_name],'feature')

    end
end



