load('data3 2.mat');
x = data(:,1:2);
% append ones for bias
x = [x ones(length(x),1)];
y = data(:,3:3);
stepSize = 1;
tolerance = 0.0001;
gd(x, y, stepSize, tolerance);