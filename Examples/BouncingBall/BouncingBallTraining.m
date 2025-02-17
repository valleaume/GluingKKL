%% Load dataset

addpath('utils', 'Examples/BouncingBall');
data = load("Data/raw-bouncing-ball-23-Jan-2025.mat");
data_3_labels = data.data_3;

% Plot the 2 classes of points

% Plot in the x space
figure(1)
clf
scatter(data_3_labels(1,data_3_labels(6,:)==1), data_3_labels(2,data_3_labels(6,:)==1), 8, 'r')
hold on
scatter(data_3_labels(1,data_3_labels(6,:)==0), data_3_labels(2,data_3_labels(6,:)==0), 8, 'b')
xlabel('x_1')
ylabel('x_2')
legend('After Jump', 'Before Jump' )

% Plot in the z space
figure(2)
clf
scatter3(data_3_labels(3, data_3_labels(6,:)==1), data_3_labels(4, data_3_labels(6,:)==1), data_3_labels(5, data_3_labels(6,:)==1), 8, 'r')
hold on
scatter3(data_3_labels(3, data_3_labels(6,:)==0), data_3_labels(4, data_3_labels(6,:)==0), data_3_labels(5, data_3_labels(6,:)==0), 8, 'b')
scatter3(data_3_labels(3, isnan(data_3_labels(6,:))), data_3_labels(4, isnan(data_3_labels(6,:))), data_3_labels(5, isnan(data_3_labels(6,:))), 5, 'black')
xlabel('z_1')
ylabel('z_2')
zlabel('z_3')
legend('After Jump', 'Before Jump', 'Nan' )



%% Train the classifier

% Remove Nan
mask = reshape(~isnan(data_3_labels(6, :)),1,[]);
fprintf( '%f% nan over %f% data points',sum(~mask), length(mask));
X_classifier = data_3_labels(3:5, mask);
Y_classifier = data_3_labels(6, mask);


% Test and train split
cv_par_t = cvpartition(Y_classifier, 'HoldOut', 0.3);

X_train_classifier = X_classifier(:, training(cv_par_t))';
Y_train_classifier = Y_classifier(training(cv_par_t))';
X_test_classifier = X_classifier(:, test(cv_par_t))';
Y_test_classifier = Y_classifier(test(cv_par_t))';


% Train a SVM separating the z variable base on its position compared to closest jump
svmModel = fitcsvm(X_train_classifier, Y_train_classifier, 'KernelFunction', 'rbf', 'Standardize', true);


% Test
% Predict on test set
Y_pred_classifier = predict(svmModel, X_test_classifier);

% Evaluate Precision
accuracy = sum(Y_pred_classifier == Y_test_classifier) / length(Y_test_classifier);
fprintf('Precision SVM : %.2f%%\n', accuracy * 100);

% Display classification errors
false_flag = ~(Y_pred_classifier == Y_test_classifier);
figure(3)
clf

pos_vel_array = data_3_labels(1:2, mask);
pos_vel_array = pos_vel_array(:, test(cv_par_t));
disp(size(pos_vel_array));
scatter(pos_vel_array(1, false_flag), pos_vel_array(2, false_flag))
xlabel('x_1')
ylabel('x_2')
title('Missclassified points')

%% Learn T_inv on half the dataset
% In order to deal with non_injectivity of T we learn T_inv only on half
% of the dataset, chosen so that T is injective in this subset. Here we train on
% points situated after a jump.
% Test and train split
mask_after = reshape(data_3_labels(8, :) == 1, 1, []); % we use the adequate label to also include the points that are before a jump but not by much
X_after = data_3_labels(3:5, mask_after)'; % We use the datascience notation X, Y, but note that here the X array consists of the z coordinates of points
Y_after = data_3_labels(1:2, mask_after)'; % Likewise, the Y array consists of the x coordinates of points

% Split into test and train set
cv_after = cvpartition(size(Y_after, 1), 'HoldOut', 0.3);
X_train_after = X_after(training(cv_after), :);
Y_train_after = Y_after(training(cv_after), :);
X_test_after = X_after(test(cv_after), :);
Y_test_after = Y_after(test(cv_after), :);

% Normalize datas
[X_train_after, mu_a, sigma_a] = zscore(X_train_after);
X_test_after = (X_test_after - mu_a) ./ sigma_a;

% Train neural network
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
    'ValidationData',{X_test_after, Y_test_after}, ...
    'ValidationFrequency',30, ...
    'MaxEpochs', 100, ...
    'InitialLearnRate', 0.01, ...
    'L2Regularization', 1e-5, ...
    'Plots', 'training-progress');

% Train the network
mdl_a = trainNetwork(X_train_after, Y_train_after, layers, options);

% Test Network
% Predict on test set
Y_pred_after = predict(mdl_a, X_test_after);

% Evaluate network performance with the RMSE metric
rmse = sqrt(mean((Y_pred_after - Y_test_after).^2, 'all'));
fprintf('RMSE after jumps : %.4f\n', rmse);

%% Learn T_inv on other half the dataset
% In order to deal with non_injectivity of T we learn T_inv only on half
% of the dataset, chosen so that T is injective in this subset. Here we train on
% points situated before a jump.

% Test and train split

mask_before = reshape(data_3_labels(7, :) == 1, 1, []); % We use the adequate label to also include the points that are after a jump but not by much


X_before = data_3_labels(3:5,mask_before)'; % We use the datascience notation X, Y, but note that here the X array consists of the z coordinates of points
Y_before = data_3_labels(1:2,mask_before)'; % Likewise, Y consists of the x coordinates of points

% Split into test and train set
cv_before = cvpartition(size(Y_before, 1), 'HoldOut', 0.3);
X_train_before = X_before(training(cv_before), :);
Y_train_before = Y_before(training(cv_before), :);
X_test_before = X_before(test(cv_before), :);
Y_test_before = Y_before(test(cv_before), :);

% Normalize datas
[X_train_before, mu_b, sigma_b] = zscore(X_train_before);
X_test_before = (X_test_before - mu_b) ./ sigma_b;
% Train Neural Network
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
    'ValidationData',{X_test_before,Y_test_before}, ...
    'ValidationFrequency',30, ...
    'MaxEpochs', 100, ...
    'InitialLearnRate', 0.01, ...
    'L2Regularization', 1e-5, ...
    'Plots', 'training-progress');

% Train network
mdl_b = trainNetwork(X_train_before, Y_train_before, layers, options);

% Test Neural Network
% Predict on test set
Y_pred_before = predict(mdl_b, X_test_before);

% Evaluate network performance with the RMSE metric
rmse = sqrt(mean((Y_pred_before - Y_test_before).^2, 'all'));
fprintf('RMSE before jumps : %.4f\n', rmse);

%% Save models

today = string(datetime("today"));
directory = 'ObserverModels/';
A = data.A;
B = data.B;
models_name = strcat(directory, 'bouncing-ball-predictor-', today, '.mat');
save(models_name, 'mdl_b', 'mdl_b', "mu_b", "sigma_b", "mdl_a", "mu_a", "sigma_a", "svmModel", 'A', 'B');