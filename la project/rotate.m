% 1. Read The Image and save it in an grayscale matrix matrix
% multiplications are possible only if we use 2-D arrays
img = rgb2gray(imread('C:\Users\Anand\Downloads\sop.png'));
% 2. Open your 'line image' as a logical array the > 0 part makes it
% logical and then convert to grayscale
line =rgb2gray( imread('C:\Users\Anand\Downloads\sob.png')) > 0;
% 3. Now, we will "align" the upper part of the image based on the line,
% so that the line will be straight at the bottom of the image. We will
% do that by sorting the 'line image', moving the ones of the mask This is
% to remove the minute faults that occur while drawing a line but its not
% exactly straight 

upper = zeros(size(line));  %creating an array 2d array of lines dimensions
upper(~line) = -1;          %comparing the line image and putting -1 in upper
                            %where black color appears
upper = sort(upper, 'descend');               %sorting 
upper(upper == -1) = img(~line); %comparing with upper if its -1 then copy 
                                    % images pixels
[m,n]= size(line);
Y= eye (m);                                %identity matrix of m dimensions 
[m,m]=size(Y);


for i=1:m/2                              %to reflect identity matrix
    temp=Y(i,:);
    Y(i,:)=Y(m-i+1,:);
	Y(m-i+1,:)=temp;
   
end
up2 = Y*upper;                    % we're using matrix transformation here
                                    %to get the mirror up2 is the mirror
imshow(up2);
% 4. We concatenate both the image with it's mirror below.
imgConcat = [upper; up2];
imgConcat=mat2gray(imgConcat);
imshow(imgConcat);
% 5. Also, The line mask will be concatenated to it's negative, and we'll
% invert the order of the rows.
lineConcat = [line; ~line];
[m,n]=size(lineConcat);
Y= eye (m);         %identity matrix
[m,m]=size(Y);
%imshow(Y);

for i=1:m/2                  %trying to reflect identity matrix
    temp=Y(i,:);                %to get a 0 matrix with the non-principal 
                                %diagonal filled with ones
    Y(i,:)=Y(m-i+1,:);
	Y(m-i+1,:)=temp;
   
end


lineConcat = Y*lineConcat; %  we're using matrix transformation here 
                           % to invert rows
lineConcat=mat2gray(lineConcat)>0; %converting to logical image
% 6. Now we repeat the "alignment procedure" used on step 4 so that the
% image will be positioned on the upper part. We will also remove the
% lower part, now containing only zeros.

mirror = zeros(size(lineConcat));  %creating the final image
mirror(lineConcat) = -1;            %putting -1s as we did earlier in upper
mirror = sort(mirror, 'ascend');   %sorting to get straight lines
mirror(mirror == -1) = imgConcat(lineConcat);% replacing the where -1s
                                            %appear with the ImgConcat
mirror = mirror(1:end/2,:,:);
% displaying the result  
 subplot(2,3,1), imshow(img, []);
 title('Step 1. Original image');
 subplot(2,3,2), imshow(double(line), []);
 title('Step 2. Separation image'); 
 subplot(2,3,3), imshow(upper, []);
 title('Step 3. Image "alignment"');
 subplot(2,3,4), imshow(imgConcat, []);
 title('Step 4. Mirror concatenation');
 subplot(2,3,5), imshow(double(lineConcat), []);
 title('Step 5. Mask concatenation');
 subplot(2,3,6), imshow(mirror, []);
 title('Step 6. Result by a final alignment');