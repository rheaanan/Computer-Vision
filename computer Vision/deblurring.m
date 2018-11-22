V=imread('D:\Program Files\matlab\toolbox\images\imdemos\trees.tif');
subplot(1,5,1)
imshow(V);
title('original image')
PSF=fspecial('Gaussian',5,5);
blurr=imfilter(V,PSF,'symmetric','conv');
subplot(1,5,2)
imshow(blurr);
title('gaussian blurred')
A=imread('D:\Program Files\matlab\toolbox\images\imdemos\trees.tif');
psf=fspecial('motion',9,90);
new=imfilter(A,psf,'symmetric','conv')
subplot(1,5,3)
imshow(new)
title('motion blurred')
I=deconvlucy(blurr,PSF,5)
subplot(1,5,4)
imshow(I);
title('deblur gaussian')
I1=deconvlucy(new,psf,5)
subplot(1,5,5)
imshow(I1)
title('deblur motion')