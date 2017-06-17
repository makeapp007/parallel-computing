function [ A ] = TXT2IMG( txtFilename )
% this function can help you to convert txt file to image and show it out.
% your program's output file should be able to be loaded by this function if the format is correct.
% txtFilename is the file name of your output
A = load(txtFilename)/255;
imshow(A);
imwrite(A,'out.jpg');
end

