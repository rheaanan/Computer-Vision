 
Y=imread('D:\Program Files\matlab\toolbox\images\imdemos\pears.png');
%imshow(Y);
i=rgb2gray(Y);
J=imnoise(i,'gaussian',0,0.025);
%imshow(i);
imshow(J);