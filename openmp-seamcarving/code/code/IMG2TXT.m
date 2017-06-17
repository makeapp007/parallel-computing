function [ A ] = IMG2TXT( imgFilename, txtFilename )
% this function can help you to convert image file to txt file which can
% make your C/C++ program read image data without any third party libraries
% imgFilename is the input image file name
% txtFilename is the output txt file name 

A = imread(imgFilename);
if size(A,3)>1
    A = rgb2gray(A);
end
[m ,n] = size(A);
fid = fopen(txtFilename,'w');
fprintf(fid,'%d %d \n',m,n);
for i = 1: m
    fprintf(fid,'%d ',A(i,:));
    fprintf(fid,'\n');
end
fclose(fid);
end