Y=imread('C:\Users\Anand\Downloads\white-hand-sketeched-star-pattern-620x400.png');
Y=rgb2gray(Y);
[m,n]=size(Y);
Z= zeros(m*2,n*2,class(Y)); 
column_inc=0;
row_inc=0;


for i=1:n       %copying the first image onto matrix a
    Z(1:m,i)=Y(:,i);
end

   
for i=1:n      %searching for the column  
    if (Y(:,n)==Y(:,i))  %if the last column of the image matches any other column
        for j=i:n           %copy the image till its boundary
            Z(1:m,n+j-i+1)=Y(:,j);
            column_inc=column_inc+1;
        end
        break
    end
end
for i=1:n      %searching for the row 
    if (Y(m,:)==Y(i,:))  %if the last row of the image matches any other row
        for j=i:m           %copy the image till its boundary
            Z(m+j-i+1,1:n)=Y(j,:);
             row_inc=row_inc+1;
            
        end
        break
    end
end
row_final=m+row_inc;            %calculations to fill in the fourth image
column_final=n+column_inc;
row_start=m-row_inc+1;
column_start=n-column_inc+1;
for i=n:column_final-1 
       row_start=m-row_inc+1;                         %attaching the fourth image  
        for j=m:row_final-1
            Z(j,i)=Y(row_start,column_start);
         row_start=row_start+1;   
            
            
        end
        column_start=column_start+1;  
     
 end

imshow(Z);


            