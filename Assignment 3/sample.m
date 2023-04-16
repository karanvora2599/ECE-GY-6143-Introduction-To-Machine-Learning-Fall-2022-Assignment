
%Loading dataset
load dataset ;
%Calculating number of rows in X
[rows, columns]=size(X);
%Changing the values of Y from 0 to (-1)
Y(Y==0)=-1;
%Normalizing the data for rbfnel
X=normalize(X);
%Randomly deciding the index value from where data is to be splitted
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
global p1;
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
[sigma_RBF,c_RBF]=find(errRbf_d==min_RBF);
