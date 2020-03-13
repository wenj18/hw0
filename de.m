nx=200;
ny=200;
lambda=50;
img=100*ones(nx,ny);
img(75:150,75:150)=150*ones(76);
nimg=img+12*randn(nx,ny);
%构造外层加一圈0的图片
img1=zeros(nx+2,ny+2);
img1(2:nx+1,2:ny+1)=img;
nimg1=zeros(nx+2,ny+2);
nimg1(2:nx+1,2:ny+1)=nimg;
output=zeros(nx+2,ny+2);
k=1;
while k<2000
    a=output;
    output(2,2)=1/(2*lambda+1)*(nimg1(2,2)+lambda*a(1,2)+lambda*a(2,1)+lambda*a(3,2)+lambda*a(2,3));
    output(2,nx+1)=1/(2*lambda+1)*(nimg1(2,nx+1)+lambda*a(1,nx+1)+lambda*a(2,nx)+lambda*a(3,nx+1)+lambda*a(2,nx+2));
    for j=3:nx
       output(2,j) =1/(3*lambda+1)*(nimg1(2,j)+lambda*a(1,j)+lambda*a(2,j-1)+lambda*a(3,j)+lambda*a(2,j+1));
    end
    for i=3:nx
        output(i,2)=1/(3*lambda+1)*(nimg1(i,2)+lambda*a(i-1,2)+lambda*a(i,1)+lambda*a(i+1,2)+lambda*a(i,3));
    end
    for i=3:nx
        output(i,ny+1)=1/(3*lambda+1)*(nimg1(i,ny+1)+lambda*a(i-1,ny+1)+lambda*a(i,ny)+lambda*a(i+1,ny+1)+lambda*a(i,ny+2));
    end
    output(nx+1,2)=1/(2*lambda+1)*(nimg1(nx+1,2)+lambda*a(nx,2)+lambda*a(nx+1,1)+lambda*a(nx+2,2)+lambda*a(nx+1,3));
    output(nx+1,ny+1)=1/(2*lambda+1)*(nimg1(nx+1,ny+1)+lambda*a(nx,ny+1)+lambda*a(nx+1,ny)+lambda*a(nx+2,ny+1)+lambda*a(nx+1,ny+2));
    for j=3:nx
       output(ny+1,j) =1/(3*lambda+1)*(nimg1(ny+1,j)+lambda*a(ny,j)+lambda*a(ny+1,j-1)+lambda*a(ny+2,j)+lambda*a(ny+1,j+1));
    end
    
for i=3:nx
    for j=3:ny
      output(i,j)=1/(4*lambda+1)*(nimg1(i,j)+lambda*a(i-1,j)+lambda*a(i,j-1)+lambda*a(i+1,j)+lambda*a(i,j+1));
    end
end
k=k+1;
end
output=output(2:nx+1,2:ny+1);
subplot(1,3,1)
imshow(img,[])
title('原图');
subplot(1,3,2)
imshow(nimg,[])
title('加Gauss噪音');
subplot(1,3,3)
imshow(output,[])
title('梯度L^2 norm去噪, \lambda=50');