clc
clearvars
function [nsv, alpha, b0] = svc(X,Y,rbf,C)
%SVC Support Vector Classification
%
% Usage: [nsv alpha bias] = svc(X,Y,rbf,C)
%
% Parameters: X - Training inputs
% Y - Training targets
% rbf - rbfnel function
% C - upper bound (non-separable case)
% nsv - number of support vectors
% alpha - Lagrange Multipliers
% b0 - bias term
%
% Author: Steve Gunn (srg@ecs.soton.ac.uk)
 if (nargin <2 || nargin>4) % check correct number of arguments
     help svc
 else 
     fprintf('Support Vector Classification\n') 
     fprintf('_____________________________\n')
     n = size(X,1);
     if (nargin<4) 
         C=Inf; 
     end
     if (nargin<3) 
         rbf='linear'; 
     end
     % tolerance for Support Vector Detection
     epsilon = svtol(C);
     % Construct the rbfnel matrix
     fprintf('Constructing ...\n');
     H = zeros(n,n);
     for i=1:n 
         for j=1:n 
             H(i,j) = Y(i)*Y(j)*svrbfnel(rbf,X(i,:),X(j,:));
         end
     end
     c = -ones(n,1);
     % Add small amount of zero order regularisation to
     % avoid problems when Hessian is badly conditioned.
     H = H+1e-10*eye(size(H));
     vlb = zeros(n,1); % Set the bounds: alphas >= 0
     vub = C*ones(n,1); % alphas <= C
     x0 = zeros(n,1); % The starting point is [0 0 0 0]
     neqcstr = nobias(rbf); % Set the number of equality constraints(1 or 0)
     if neqcstr
         A = Y'; b = 0; % Set the constraint Ax = b
     else
         A = []; b = [];
     end
    
     % Solve the Optimisation Problem

 
     fprintf('Optimising ...\n');
     st = cputime;
     % [alpha lambda how] = quadprog(H, c, A, b, [],[], vlb,vub, x0);
     [alpha , ~, how] = quadprog(H, c, [],[],A, b, vlb, vub, x0);
     fprintf('Execution time: %4.1f seconds\n',cputime -st) 
     fprintf('Status : %s\n',how); 
     w2 = alpha'*H*alpha; 
     fprintf('|w0|^2 : %f\n',w2); 
     fprintf('Margin : %f\n',2/sqrt(w2)); 
     fprintf('Sum alpha : %f\n',sum(alpha));
     % Compute the number of Support Vectors
     svi = find( alpha > epsilon);
     nsv = length(svi);
     fprintf('Support Vectors : %d%3.1f%%)\n',nsv,100*nsv/n);
     b0 = 0;
     if nobias(rbf) ~= 0
      svii = find( alpha > epsilon & alpha < (C -epsilon));
      if ~isempty(svii)
          b0 = (1/length(svii))*sum(Y(svii) -H(svii,svi)*alpha(svi).*Y(svii));
      else
          fprintf('No support vectors on margin - cannot compute bias.\n');
      end
     end
 end

function err = svcerror(trnX,trnY,tstX,tstY,rbf,alpha,bias)
%SVCERROR Calculate SVC Error
%
% Usage: err =svcerror(trnX,trnY,tstX,tstY,rbf,alpha,bias)
%
% Parameters: trnX - Training inputs
% trnY - Training targets
% tstX - Test inputs
% tstY - Test targets
% rbf - rbfnel function
% beta - Lagrange Multipliers
% bias - bias
%
% Author: Steve Gunn (srg@ecs.soton.ac.uk)
 if (nargin ~= 7) 
     help svcerror
 else
     n = size(trnX,1);
     m = length(tstY);
     H = zeros(m,n);
     for i=1:m
         for j=1:n
             H(i,j) = trnY(j)*svrbfnel(rbf,tstX(i,:),trnX(j,:));
         end
     end
     predictedY = sign(H*alpha + bias);
     err = sum(predictedY ~= tstY);
 end



%Loading dataset
load dataset;
%Calculating number of rows in X
[rows,columns]=size(X);
%Changing the values of Y from 0 to (-1)
Y(Y==0)=-1;
%Normalizing the data for rbfnel
X=normalize(X);
%Randomly deciding the index value from where data is to be
splitted
indexTrain = randsample (rows, rows/2);
%Splitting the dataset into train and test
indexTest = setdiff (1:rows, indexTrain );
trainX = X(indexTrain,:);
trainY= Y(indexTrain,:);
testX= X(indexTest,:);
testY= Y(indexTest,:);
%Linear rbfnel
cMax=20;
errLinear_d = zeros(1 , cMax);
cList=zeros(1,cMax);
for i=1:cMax
 if i<=8
     c = 0.1^(8-i);
 else
     c = (c+1);
 end

 %Train error for linear rbfnel
 [nsv,alpha,bias] = svc(trainX,trainY,'linear',c);

 %Test error for linear rbfnel
 errLinear_d(i)=svcerror(trainX,trainY,testX,testY,'linear',alpha,bias);
 cList(i)=c;

end
%Plotting linear rbfnel
f = figure(1);
clf(f);
plot(1:20, errLinear_d);
xticks(1:20)
xticklabels(cList)
[~,~] = min(errLinear_d);
title('Performance of linear rbfnel');
xlabel('Error');
ylabel('Number of misclassifications')
%Polynomial rbfnel
degreeMax=20;
errPolynomial_d = zeros(degreeMax,cMax);
degree_list=zeros(1,degreeMax);
for d=1:degreeMax
 p1 = d;
 degree_list(d)=p1;
 for i=1:cMax
 if i<=8
 c = 0.1^(8-i);
 else
 c = (c+1);
 end
 %Training error for polynomial rbfnel
 [nsv , alpha , bias ] = svc (trainX , trainY ,'poly' , c );
 %Testing error for polynomial rbfnel
 errPolynomial_d(d,i)= svcerror(trainX , trainY ,testX , testY , 'poly' ,alpha , bias );
 end
end
%Plotting polynomial rbfnel
f = figure(2);
clf(f);
bar3(errPolynomial_d);
set(gca,'YTickLabel',degree_list);
set(gca,'XTickLabel',cList);
title('Performance of Polynomial rbfnal');
xlabel('Error');
ylabel('Degree of polynomial');
zlabel('Number of misclassifications')
min_Poly=min(min(errPolynomial_d));
[~,~]=find(errPolynomial_d==min_Poly);
%RBF rbfnel
sigma_max=20;
errRbf_d=zeros(cMax , sigma_max);
sigma_list=zeros(1,sigma_max);
for sigma=1:sigma_max
 p1 = 1+0.5*(sigma-1);
 sigma_list(sigma)=p1;
 for i=1:cMax
 if i<=8
 c = 0.1^(8-i);
 else
 c = (c+1);
 end
 %Train error for RBF rbfnel
 [nsv , alpha , bias ] = svc (trainX , trainY ,'rbf' , c );
 %Test error for RBF rbfnel
 errRbf_d(sigma,i)= svcerror(trainX , trainY , testX, testY , 'rbf' ,alpha , bias );
 end
end
%Plotting performance of RBF rbfnel
f = figure(3);
clf(f);
bar3(errRbf_d)
set(gca,'YTickLabel',sigma_list);
set(gca,'XTickLabel',cList);
title('Performance of RBF rbfnal');
xlabel('Error');
ylabel('Sigma')
zlabel('Number of misclassfications')
min_RBF=min(min(errRbf_d));
[~,~]=find(errRbf_d==min_RBF);
end
end
