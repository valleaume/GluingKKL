%% Load dataset

data = load("Data/raw-bouncing-ball-23-Jan-2025.mat");

%% Plot the 2 classes of points


% Plot in the x space
figure(3)
clf
scatter(data(1,data(6,:)==1), data(2,data(6,:)==1), 8, 'r')
hold on
scatter(data(1,data(6,:)==0), data(2,data(6,:)==0), 8, 'b')
xlabel('x_1')
ylabel('x_2')
scatter(Init_conditions(1,:), Init_conditions(2,:), 4)
legend('After Jump', 'Before Jump', 'Init Conditions' )

% Plot in the z space
figure(4)
clf
scatter3(data(3, data(6,:)==1), data(4, data(6,:)==1), data(5, data(6,:)==1), 8, 'r')
hold on
scatter3(data(3, data(6,:)==0), data(4, data(6,:)==0), data(5, data(6,:)==0), 8, 'b')
scatter3(data(3, isnan(data(6,:))), data(4, isnan(data(6,:))), data(5, isnan(data(6,:))), 5, 'black')
xlabel('z_1')
ylabel('z_2')
zlabel('z_3')
legend('After Jump', 'Before Jump', 'Nan' )

%% Remove Nan
mask = reshape(~isnan(data(6, :)),1,[]);
disp(sum(~mask));
X = data(3:5, mask);
Y = data(6, mask);

%% Train the classifier
% 2 models ared tested : svm with radial kernels and random forest

%% Test and train split
cv_par_t = cvpartition(Y, 'HoldOut', 0.3);
disp(cv_par_t);

XTrain = X(:, training(cv_par_t))';
YTrain = Y(training(cv_par_t))';
XTest = X(:, test(cv_par_t))';
YTest = Y(test(cv_par_t))';


%% Training model(s)
% Train a randomForest classifier
nTrees = 100; % number of tree in the forest


% Train the forest
randomForest = TreeBagger(nTrees, XTrain, YTrain, 'Method', 'classification');

svmModel = fitcsvm(XTrain, YTrain, 'KernelFunction', 'rbf', 'Standardize', true);


%% Test
% Predict on test set
YPred = predict(svmModel, XTest);

% Evaluate Precision
accuracy = sum(YPred == YTest) / length(YTest);
fprintf('Precision SVM : %.2f%%\n', accuracy * 100);



YPred = predict(randomForest, XTest);
YPred = str2double(YPred); % convert into numbers

% Evaluate Precision
accuracy = sum(YPred == YTest) / length(YTest);
fprintf('Precision Forest : %.2f%%\n', accuracy * 100);
%% Display classification errors
false_flag = ~(YPred == YTest);
figure(5)
clf
%scatter3(XTest(false_flag, 1), XTest(false_flag, 2), XTest(false_flag, 3))
X_x = data(1:2, mask);
X_x = X_x(:, test(cv_par_t));
disp(size(X_x));
scatter(X_x(1, false_flag), X_x(2, false_flag))
xlabel('x_1')
ylabel('x_2')
title('Missclassified points')

%% Learn T_inv on half the dataset
% In order to deal with non_injectivity of T we learn T_inv only on half
% of the dataset, chosen so that T is injective in this subset. Here we train on
% points situated after a jump.
%% Test and train split
mask_a = reshape(data(8, :)==1, 1, []);
X_a = data(3:5, mask_a)';
Y_a = data(1:2, mask_a)';

% Split into test and train set
cv = cvpartition(size(Y_a, 1), 'HoldOut', 0.3);
XTrain_a = X_a(training(cv), :);
YTrain_a = Y_a(training(cv), :);
XTest_a = X_a(test(cv), :);
YTest_a = Y_a(test(cv), :);

% Normalize datas
[XTrain_a, mu_a, sigma_a] = zscore(XTrain_a);
XTest_a = (XTest_a - mu_a) ./ sigma_a;

%% Train neural network
% Create neural network
layers = [
    featureInputLayer(3)
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(2)
    regressionLayer];

% Training options
options = trainingOptions('adam', ...
    'ValidationData',{XTest_a,YTest_a}, ...
    'ValidationFrequency',30, ...
    'MaxEpochs', 100, ...
    'InitialLearnRate', 0.01, ...
    'L2Regularization', 1e-5, ...
    'Plots', 'training-progress');

% Train the network
mdl_a = trainNetwork(XTrain_a, YTrain_a, layers, options);

%% Test Network
% Predict on test set
YPred_a = predict(mdl_a, XTest_a);

% Evaluate network performance with the RMSE metric
rmse = sqrt(mean((YPred_a - YTest_a).^2, 'all'));
fprintf('RMSE : %.4f\n', rmse);

%% Learn T_inv on other half the dataset
% In order to deal with non_injectivity of T we learn T_inv only on half
% of the dataset, chosen so that T is injective in this subset. Here we train on
% points situated before a jump.
%% Test and train split

mask_b = reshape(data(7, :) == 1, 1, []);


X_b = data(3:5,mask_b)';
Y_b = data(1:2,mask_b)';

% Split into test and train set
cv = cvpartition(size(Y_b, 1), 'HoldOut', 0.3);
XTrain_b = X_b(training(cv), :);
YTrain_b = Y_b(training(cv), :);
XTest_b = X_b(test(cv), :);
YTest_b = Y_b(test(cv), :);

% Normalize datas
[XTrain_b, mu_b, sigma_b] = zscore(XTrain_b);
XTest_b = (XTest_b - mu_b) ./ sigma_b;

%% Train Neural Network
% Create neural network
layers = [
    featureInputLayer(3)
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(2) 
    regressionLayer];

% Training options
options = trainingOptions('adam', ...
    'ValidationData',{XTest_b,YTest_b}, ...
    'ValidationFrequency',30, ...
    'MaxEpochs', 100, ...
    'InitialLearnRate', 0.01, ...
    'L2Regularization', 1e-5, ...
    'Plots', 'training-progress');

% Train network
mdl_b = trainNetwork(XTrain_b, YTrain_b, layers, options);

%% Test Neural Network
% Predict on test set
YPred_b = predict(mdl_b, XTest_b);

% Evaluate network performance with the RMSE metric
rmse = sqrt(mean((YPred_b - YTest_b).^2, 'all'));
fprintf('RMSE : %.4f\n', rmse);

%% Save models

today = string(datetime("today"));
directory = 'ObserverModels/';
models_name = strcat(directory, 'bouncing-ball-predictor-', today, '.mat');
save(models_name, 'mdl_b', 'mdl_b', "mu_b", "sigma_b", "mdl_a", "mu_a", "sigma_a", "svmModel", "randomForest", 'A', 'B');