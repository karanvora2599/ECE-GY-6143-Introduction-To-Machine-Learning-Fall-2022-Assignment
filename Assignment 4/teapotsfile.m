load('teapots.mat')
data = teapotImages;
u = mean(data);
figure(13);
colormap 'gray';
subplot(1,1,1);
imagesc(reshape(v(:,3),38,50));
title(d(3));
axis image;
X = data - u;
C = cov(X);
[Var, D] = eig(C);
[dg, ind] = sort(diag(D),'descend');
dg = dg(1:3,:);
fprintf('Eigen Value = %f\n', dg);
v = Var(:,ind(1:3));
c = X*v;
X_hat = u+c*v';
for i = 1:10
    figure(i);
    colormap gray;
    subplot(1,2,1);
    imagesc(reshape(data(i,:),38,50));
    title('Original image');
    axis image;
    subplot(1,2,2)
    imagesc(reshape(X_hat(i,:),38,50));
    title('PCA image');
    axis image;

end

norm(data-X_hat)