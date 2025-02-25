%% Load existing dataset (or run BouncingBallDataGeneration to create a new one)

addpath('utils', 'Examples/StickSlip'); 
close all; % close all previously opened figures

data = load("Data/raw-stick-slip.mat");
data_3_labels = data.data_3;

% Dataset structure :
% 1:5 = x
% 6:11 = z
% 12 = label 1 if "after jump", label 0 if "before jump", Nan if not enough jump during trajectory
% 13 = label 1 if "after jump" or close to "after jump" ; 0 otherwise
% 14 = label 1 if "before jump" or close to "before jump" ; 0 otherwise

% In all this file, we use the datascience convention where X (resp. Y)
% denotes the inputs (resp. outputs) of NN models
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




%% Plot the points x in dataset, depending on whether they are labelled "before jump" or "after jump"

% Plot the 2 classes of points

mask_after_jump = data_3_labels(aug_sys.state_dimension + 1, :)==1 ; % to select points labelled as "after jump"
mask_before_jump = data_3_labels(aug_sys.state_dimension + 1, :)==0 ; % to select points labelled as "before jump"
mask_nan = (isnan(data_3_labels(aug_sys.state_dimension + 1, :))); % to select points labelled as "NAN"

% Plot in the (position, velocity) space
figure(1)
clf
scatter(data_3_labels(1, mask_after_jump), data_3_labels(2, mask_after_jump), 8, 'r')
hold on
scatter(data_3_labels(1, mask_before_jump), data_3_labels(2, mask_before_jump), 8, 'b')
xlabel('$x_1$', Interpreter='latex')
ylabel('$x_2$', Interpreter='latex')
legend('After Jump', 'Before Jump', Interpreter='latex')
title('Dataset in $x_1, x_2$-coordinates', Interpreter='latex')
grid on

% Plot in the (x, v, mu_s ) space
figure(2)
clf
scatter3(data_3_labels(1, mask_after_jump), data_3_labels(2, mask_after_jump), data_3_labels(3, mask_after_jump), 8, 'r')
hold on
scatter3(data_3_labels(1, mask_before_jump), data_3_labels(2, mask_before_jump), data_3_labels(3, mask_before_jump), 8, 'b')
scatter3(data_3_labels(1, mask_nan), data_3_labels(2, mask_nan), data_3_labels(3, mask_nan), 4, 'black')
xlabel('$x_1$', Interpreter='latex')
ylabel('$x_2$', Interpreter='latex')
zlabel('$\mu_s$', Interpreter='latex')
title('Dataset in $x_1, x_2, \mu_s$-coordinates', Interpreter='latex')
legend('After Jump', 'Before Jump', 'Nan', Interpreter='latex')

%% Train the classifier
% This classifier should learn to recognize whether a given z corresponds
% to a point x labelled "before jump" or "after jump" 
% => input z and output "before/after jump"

% Remove Nan and define the classifier input X and output Y
mask = reshape(~isnan(data_3_labels(aug_sys.state_dimension + 1, :)), 1, []);
fprintf( '%f% nan over %f% data points',sum(~mask), length(mask));
X_classifier = data_3_labels(aug_sys.nx + 1:aug_sys.nx+aug_sys.nz, mask); % z component
Y_classifier = data_3_labels(aug_sys.state_dimension + 1, mask); % "after/before jump" label


% Test and train split
cv_par_t = cvpartition(Y_classifier, 'HoldOut', 0.3);

X_train_classifier = X_classifier(:, training(cv_par_t))';
Y_train_classifier = Y_classifier(training(cv_par_t))';
X_test_classifier = X_classifier(:, test(cv_par_t))';
Y_test_classifier = Y_classifier(test(cv_par_t))';


% Train a SVM predicting whether a z-state correspond to a x-state that is "after" or "before" a jump 
svmModel = fitcsvm(X_train_classifier, Y_train_classifier, 'KernelFunction', 'rbf', 'Standardize', true);


% Test
% Predict on test set
Y_pred_classifier = predict(svmModel, X_test_classifier);

% Evaluate Precision
accuracy = sum(Y_pred_classifier == Y_test_classifier) / length(Y_test_classifier);
fprintf('Precision SVM : %.2f%%\n', accuracy * 100);

% Display classification errors
false_flag = ~(Y_pred_classifier == Y_test_classifier);  % indices in the test data where the classifier made mistakes
figure(3)
clf

x_test = data_3_labels(1:2, mask);
x_test = x_test(:, test(cv_par_t)); % x component corresponding to the z-components used for testing the classifier
disp(size(x_test));
scatter(x_test(1, false_flag), x_test(2, false_flag))
xlabel('$x_1$', Interpreter='latex')
ylabel('$x_2$', Interpreter='latex')
title('Missclassified points')

%%%% Other possibility %%%%

% A random forest also exhibited good results
% Train a randomForest classifier

%nTrees = 100; % number of tree in the forest
%randomForest = TreeBagger(nTrees, X_train_classifier, Y_train_classifier, 'Method', 'classification');

%YPred = predict(randomForest, XTest);
%YPred = str2double(YPred);    % for a randomForest you need to convert the output into numbers

%% Learn T_inv on half the dataset
% => input z and output x
% In order to deal with non_injectivity of T we learn T_inv only on half
% of the dataset, chosen so that T is injective in this subset. Here we train on
% points classified as "after jump". To improve generalization and avoid errors
% at the boundary between the "before/after jumps" labels, we also include
% points classified as "before jump" but very close to the boundary

% Test and train split
mask_after = reshape(data_3_labels(aug_sys.state_dimension + 3, :) == 1, 1, []);  % we use the adequate label to also include the points that are before a jump but not by much
X_after = data_3_labels(aug_sys.nx + 1:aug_sys.nx + aug_sys.nz, mask_after)'; % z component, input
Y_after = data_3_labels(1:aug_sys.nx, mask_after)'; % x component, output

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

%% Learn T_inv on other half of the dataset
% => input z and output x
% In order to deal with non_injectivity of T we learn T_inv only on half
% of the dataset, chosen so that T is injective in this subset. Here we train on
% points classified as "before jump" or close to "before jump".

% Test and train split

mask_before = reshape(data_3_labels(aug_sys.state_dimension + 2, :) == 1, 1, []);  % We use the adequate label to also include the points that are after a jump but not by much


X_before = data_3_labels(aug_sys.nx + 1: aug_sys.nx + aug_sys.nz, mask_before)'; % z component, input
Y_before = data_3_labels(1:aug_sys.nx, mask_before)'; % x component, output

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