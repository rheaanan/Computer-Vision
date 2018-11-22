Y=imread('D:\Program Files\matlab\toolbox\images\imdemos\cameraman.tif');

[m,n]=size(Y);
%imshow(Y);

for i=1:n/2
    temp=Y(i,:);
    Y(i,:)=Y(n-i+1,:);
	Y(n-i+1,:)=temp;
   
end
imshow(Y);










