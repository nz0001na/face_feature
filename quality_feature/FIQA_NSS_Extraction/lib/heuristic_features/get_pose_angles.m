function [euler_x, euler_y, euler_z] =  get_pose_angles(file_path)
    pose_path = '/media/guo/My Passport/ROOTDIR/FQDB/Frontals/0_pose';
    [path, name, ~] = fileparts(file_path);
    name = name(1:end-2); % removing '_1' etc. from end of the file name
    parts = strsplit(path, '0_originals');
    pts_file_path = [pose_path '/' parts{2} '/' name '*.pts'];
    files = dir(pts_file_path);
    if length(files) > 1
        idx = randsample(length(files),1);
        pts_file_path = [pose_path '/' parts{2} '/' files(idx).name];
    elseif length(files) == 1
        pts_file_path = [pose_path '/' parts{2} '/' files(1).name];
    else % not found
        euler_x = intmax;
        euler_y = intmax;
        euler_z = intmax;
        return;
    end
    fid = fopen(pts_file_path, 'r');    
    C = textscan(fid, '%s','delimiter', '\n');
    fclose(fid);
    parts = strsplit(C{1}{75}, ' ');    
    euler_x = str2double(parts{1});
    euler_y = str2double(parts{2});
    euler_z = str2double(parts{3});
end