% Linear Regression
% We create a contrived dataset and find the least 
% squares approximation

rangeData = [-10 35];
featDim = 1;
dataNum = 150;
trainNum = 120;

rng(1);
data = randi(rangeData, 2, dataNum); 

w = rand(featDim,1);
b = rand(1);
T = 3000;
stepSize = 1e-3;

x = data(1,:);
y = data(2,:);

xTrain = x(1:trainNum);
yTrain = y(1:trainNum);

% Go through the GD 
for i = 1:T
    output = w'*xTrain + b;
    loss = (1/trainNum)*sum((yTrain - output).^2);
    gradW = -(2/trainNum)*(yTrain-output)*xTrain';
    gradB = -(2/trainNum)*sum(yTrain-output);
    w = w - stepSize*(gradW);
    b = b - stepSize*(gradB);
end

% Plot the data and the linear model
figure(1)
scatter(data(1,:), data(2,:), 'ro', 'LineWidth', 0.5, 'MarkerFaceColor', 'b')
title('Original Dataset')
plotRange = [rangeData(1)-5, rangeData(2)+5];
axis([plotRange plotRange])
hold on
model = w*plotRange + b;
% w, b
plot(plotRange, model)
hold off

% Test the slope and intercept on the test set
xTest = x(trainNum+1:end);
yTest = y(trainNum+1:end);

yPred = w*xTest + b;
acc = accuracy(yTest, yPred)


function [acc] = accuracy(y, yPred)
    sumAcc = 0;
    if (length(y) == length(yPred))
        for i = 1:length(y)
            if (y(i) == yPred(i))
                sumAcc = sumAcc+1;
            end
        end
    end
    acc = (sumAcc/length(y)) * 100;
end