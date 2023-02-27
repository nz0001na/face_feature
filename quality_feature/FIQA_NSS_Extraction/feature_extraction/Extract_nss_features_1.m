% this code is to extract nss feature from bona fide and morphed faces
clear;clc;
% add path to external functions
addpath('../lib/redist-source');
addpath('../lib/bovik/SSEQ_release/SSEQ');
addpath('../lib/bovik/BLIINDS2_release/Bliinds2_code');
addpath('../lib/bovik/DIIVINE_release/');
addpath('../lib/bovik/DIIVINE_release/sepspyr/deps/matlabPyrTools-1.3');
addpath('../lib/bovik/DIIVINE_release/sepspyr/deps/matlabPyrTools-1.3/MEX');
addpath('../lib/bovik/BRISQUE_release/BRISQUE_release');
addpath('../lib/bovik/CurveletQA_release/');
addpath('../lib/bovik/CurveletQA_release/fdct_usfft_matlab');
addpath('../lib/bovik/CurveletQA_release/fdct_usfft_matlab/CurveCoeff');
addpath('../lib/bovik/CurveletQA_release/fdct_usfft_matlab/Utilities');
addpath('../lib/bovik/CurveletQA_release/fdct_usfft_matlab/Windows/Meyer');
addpath('../lib/bovik/CurveletQA_release/fdct_usfft_matlab/Windows/IteratedSine');
addpath('../lib/bovik/CurveletQA_release/fdct_usfft_matlab/USFFT');
addpath('../lib/bovik/niqe_release');
addpath('../lib/bovik/tmiqa_release');

% 
% sseqf = {}; bliindsf = {}; diivinef = {}; brisquef = {}; curveletf = {}; niqef = {}; tmiqa = {}; 

src_path = 'D:\2_zn_research\1_MAD\Data\3_AMSL_FFHQ_cropped_grayscale_resize\size_270_270/'
dst_path = 'D:\2_zn_research\1_MAD\Data\8_NSS_feature/'

fim1 = 'londondb_genuine_neutral_passport-scale_15kb'
fim2 = 'londondb_genuine_smiling_passport-scale_15kb'
fim3 = 'londondb_morph_combined_alpha0.5_passport-scale_15kb'
fim4 = 'FFHQ'
Folder(1).name = fim1;  
Folder(2).name = fim2; 
Folder(3).name = fim3;
Folder(4).name = fim4;

f_item = {'1_sseq','2_brisque','5_curvelet','6_niqe','7_tmiqa', '3_bliinds','4_diivine'};
for n=6:length(f_item)
    item = f_item{n};
    if ~exist([dst_path item '/'], 'dir')
        mkdir([dst_path item '/'])
    end
    
    for i=1:4
        folder_path = [src_path Folder(i).name]

        if ~exist([dst_path item '/' Folder(i).name], 'dir')
            mkdir([dst_path item '/' Folder(i).name])
        end
        % ?????? .jpg
        Images = dir([folder_path, '\*.png']);
        len = size(Images,1)
        if len == 0
            continue
        end
        for j=1:len
            img_name = Images(j).name
            a = strsplit(img_name,'.')
            new_name = [a{1} '.mat']
            
            if exist([dst_path item '/' Folder(i).name '/' new_name ], 'file')
                continue
            end
            img = [src_path Folder(i).name '/' img_name]
            feature = call_feature_function_zn(img,n)         
            save([dst_path item '/' Folder(i).name '\' new_name],'feature')
    
        end
    end
end
    
    