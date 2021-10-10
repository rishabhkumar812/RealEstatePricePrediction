clear ; close all; clc

% Load CSV files
X = csvread('Real estate.csv');


% Split labels from features
y = X(2:end,8);
x = X(2:end,2:7);

trainSize = floor(size(y,1)*0.7);

% Spliting data for training and validation
 xtrain = x(1:trainSize,:);
 ytrain = y(1:trainSize);
 xtest = x(trainSize+1:end,:);
 ytest = y(trainSize+1:end);
 
% Saving to a binary zipped mat file
save -binary -zip gs_digits.mat x y xtrain ytrain xtest ytest;

%Feature Scaling and normalizing
[xtrain,muTrain,sigmaTrain]=normalize(xtrain);
[xtest,muTest,sigmaTes]=normalize(xtest);

%Adding bias column
xtrain = [ones(size(xtrain,1),1), xtrain];
xtest = [ones(size(xtest,1),1), xtest];

%Displaying 10 rows of Xtrain
[x(1:10,:), y(1:10,:)]
[xtrain(1:10,:), ytrain(1:10,:)]
[xtest(1:10,:), ytest(1:10,:)]

%Initial theta 
theta = zeros(size(xtrain,2),1);
initialCost = calcCostFunction(xtrain,theta,ytrain);
%Displaying initialCost and theta
theta
initialCost

[theta,recordJ] = doGD(theta,xtrain,ytrain);

%theta after GD
theta

%final Cost
finalCost = calcCostFunction(xtrain,theta,ytrain)

figure (1);
plot(recordJ,'r','linewidth',2);
xlabel('Iterations');
ylabel('Cost');
title('Running GD');

[cost] = evaluate(xtest,ytest,theta);

prediction = calcH(theta,xtest);

prediction = [prediction,ytest];

prediction(1:50,:);
cost




