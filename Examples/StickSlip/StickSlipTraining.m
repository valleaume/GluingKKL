%% Load dataset

addpath('utils', 'Examples/StickSlip'); 

data = load("Data/raw-stick-slip.mat");
data_3_labels = data.data_3;

% Create a StickSlip object.
sys = StickSlipSystemClass();

% Define the observation function y = h(x, t)
h = @(x, t) (x(1));

% Create the associated StickSlip object
obs_sys = ObservedHybridSystem(sys, 1, h);

% Define the AugmentedSystem
A = data.A;
B = data.B;
aug_sys = AugmentedSystem(obs_sys, 6, A, B);


% Plot the 2 classes of points

% Plot in the (position, velocity) space
figure(1)
clf
scatter(data_3_labels(1, data_3_labels(aug_sys.state_dimension + 1,:)==1), data_3_labels(2, data_3_labels(aug_sys.state_dimension + 1,:)==1), 8, 'r')
hold on
scatter(data_3_labels(1, data_3_labels(aug_sys.state_dimension + 1,:)==0), data_3_labels(2, data_3_labels(aug_sys.state_dimension + 1,:)==0), 8, 'b')
xlabel('position')
ylabel('speed')
title( 'Labelled points for multiple values of friction coefficients' )

legend('After Jump', 'Before Jump')

% Plot in the (x, v, mu_s ) space
figure(2)
clf
scatter3(data_3_labels(1, data_3_labels(aug_sys.state_dimension + 1,:)==1), data_3_labels(2, data_3_labels(aug_sys.state_dimension + 1,:)==1), data_3_labels(3, data_3_labels(aug_sys.state_dimension + 1,:)==1), 8, 'r')
hold on
scatter3(data_3_labels(1, data_3_labels(aug_sys.state_dimension + 1,:)==0), data_3_labels(2, data_3_labels(aug_sys.state_dimension + 1,:)==0), data_3_labels(3, data_3_labels(aug_sys.state_dimension + 1,:)==0), 8, 'b')
scatter3(data_3_labels(3, isnan(data_3_labels(aug_sys.state_dimension + 1,:))), data_3_labels(4, isnan(data_3_labels(aug_sys.state_dimension + 1,:))), data_3_labels(5, isnan(data_3_labels(aug_sys.state_dimension + 1,:))), 4, 'black')
xlabel('x')
ylabel('v')
zlabel('\mu_s')
title('Labeled points')
legend('After Jump', 'Before Jump', 'Nan' )

%% Train the classifier

% Remove Nan
mask = reshape(~isnan(data_3_labels(aug_sys.state_dimension + 1, :)), 1, []);
fprintf( '%f% nan over %f% data points',sum(~mask), length(mask));
X_classifier = data_3_labels(aug_sys.nx + 1:aug_sys.nx+aug_sys.nz, mask);
Y_classifier = data_3_labels(aug_sys.state_dimension + 1, mask);


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
xlabel('x')
ylabel('v')
title('Missclassified points')

%%%% Other possibility %%%%

% A random forest also exhibited good results
% Train a randomForest classifier

%nTrees = 100; % number of tree in the forest
%randomForest = TreeBagger(nTrees, X_train_classifier, Y_train_classifier, 'Method', 'classification');

%YPred = predict(randomForest, XTest);
%YPred = str2double(YPred);    % for a randomForest you need to convert the output into numbers

%% Learn T_inv on half the dataset
% In order to deal with non_injectivity of T we learn T_inv only on half
% of the dataset, chosen so that T is injective in this subset. Here we train on
% points situated after a jump.

% Test and train split
mask_after = reshape(data_3_labels(aug_sys.state_dimension + 3, :) == 1, 1, []);  % we use the adequate label to also include the points that are before a jump but not by much
X_after = data_3_labels(aug_sys.nx + 1:aug_sys.nx + aug_sys.nz, mask_after)'; % We use the datascience notation X, Y, but note that here the X array consists of the z coordinates of points
Y_after = data_3_labels(1:aug_sys.nx, mask_after)'; % Likewise, the Y array consists of the x coordinates of points

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
    featureInputLayer(aug_sys.nz)
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(aug_sys.nx) 
    regressionLayer];

cv_val_after = cvpartition(size(Y_test_after, 1), 'HoldOut', 0.3);
X_val_after = X_test_after(test(cv_val_after), :);
Y_val_after = Y_test_after(test(cv_val_after), :);

% Training options
options = trainingOptions('adam', ...
    'ValidationData',{X_val_after, Y_val_after}, ...
    'ValidationFrequency',30, ...
    'MaxEpochs', 100, ...
    'ValidationPatience', 300, ...
    'ObjectiveMetricName',"loss", ...
    'OutputNetwork', 'best-validation', ...
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

mask_before = reshape(data_3_labels(aug_sys.state_dimension + 2, :) == 1, 1, []);  % We use the adequate label to also include the points that are after a jump but not by much


X_before = data_3_labels(aug_sys.nx + 1: aug_sys.nx + aug_sys.nz, mask_before)'; % We use the datascience notation X, Y, but note that here the X array consists of the z coordinates of points
Y_before = data_3_labels(1:aug_sys.nx, mask_before)'; % Likewise, Y consists of the x coordinates of points

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
    featureInputLayer(aug_sys.nz)
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(aug_sys.nx)
    regressionLayer];

%Use some validation data as a stopping criterion
cv_val_before = cvpartition(size(Y_test_before, 1), 'HoldOut', 0.3);
X_val_before = X_test_before(test(cv_val_before), :);
Y_val_before = Y_test_before(test(cv_val_before), :);

% Training options
options = trainingOptions('adam', ...
    'ValidationData',{X_val_before,Y_val_before}, ...
    'ValidationFrequency',30, ...
    'ValidationPatience', 300, ...
    'ObjectiveMetricName',"loss", ...
    'OutputNetwork', 'best-validation', ...
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
models_name = strcat(directory, 'stick-slip-predictor-', today, '.mat');
save(models_name, 'mdl_b', 'mdl_b', "mu_b", "sigma_b", "mdl_a", "mu_a", "sigma_a", "svmModel", 'A', 'B');