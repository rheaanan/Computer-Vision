A=imread('D:\Program Files\matlab\toolbox\images\imdemos\cameraman.tif');
[m,n]=size(A);

a=zeros(m,n*2+10,class(X));
for i=1:n
    a(:,i)=A(:,i);
    
end
for i=1:n
    a(:,n+10+i)=A(:,i);
    
end
for i=1 :m
    l=A(i,n);
    r=A(i,1);
    if(l>r)
        diff=l-r;
        step=diff/10;
        for j=1:10
            a(i,n+j)=l;
            l=l-step;
        end
    else
        diff=r-l;
        step=diff/10;
        for j=1:10
            a(i,n+j)=l;
            l=l+step;
        end
    end
end

imshow(a);
