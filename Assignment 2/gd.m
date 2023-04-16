function [] = gd(x, y, stepSize, tolerance)
% get dataset dimensions
[n, m] = size(x);
% initialize weight vectors
theta = rand(m, 1);
thetaPrev = zeros(m, 1);
% graph data
iter = 1;
perceptronErr = [];
binClassErr = [];
errPlotX = [];
while true
    % binary classification error
    bErr = 0;
    % perceptron error
    pErr = 0;
    % gradient
    g = 0;
    % predicted value by model
    pred = 1./(1+exp(-x*theta));
    for i=1:n
        % classify prediction
        if (pred(i) < 0.5)
            activated = -1;
        else
            activated = 1;
        end
        % check for misclassification
        if (activated ~= y(i))
            % update gradient
            g = g + (x(i,:)' * y(i));
            % update errors
            pErr = pErr + y(i)*(x(i,:)*theta);
            bErr = bErr + 1;
        end 
    end
    
    % complete calculation of gradient,errors
    bErr = (1/n) * bErr;
    pErr = (-1/n) * pErr;
    g = (-1/n) * g;
    
    % store data for plots
    perceptronErr(iter) = pErr;
    binClassErr(iter) = bErr;
    errPlotX(iter) = iter;
    disp("iter: " + iter + " perceptronErr: " + pErr + " binClassErr: " + bErr);
    % calculate update weights
    theta = thetaPrev - stepSize*g;
    if (bErr==0 || tolerance > norm(theta-thetaPrev)) 
        break;
    end

    thetaPrev = theta;
    iter = iter + 1;
end
% plot linear regression groups and boundary
figure
% plot linear regression groups
plot(x(:,1), x(:,2), 'bX');
hold on;
disp("iteration count = " + iter);

% plot linear decision boundary
xBoundary = 0:0.01:1;
yBoundary = -(theta(1)/theta(2))*xBoundary - (theta(3)/theta(2)); plot(xBoundary, yBoundary, 'r--');
hold off
legend('Data Points', 'Linear Decision Boundary');
% plot errors
figure
% plot empirical errors
plot(errPlotX, perceptronErr, 'r--'); hold on
% plot binary classification errors
plot(errPlotX, binClassErr, 'b--');
hold off
legend('Perception Error', 'Binary Classification Error');
end