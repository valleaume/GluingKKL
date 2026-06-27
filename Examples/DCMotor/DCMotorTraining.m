%% Load the latest DC Motor dataset and train an inverse gluing predictor

addpath('utils', 'Examples/DCMotor');
close all;

% Find the latest generated DC motor dataset.
files = dir('Data/raw-dc-motor-*.mat');
assert(~isempty(files), 'No raw DC motor dataset found in Data/');
[~, idx] = max([files.datenum]);
dataset_name = files(idx).name;
disp(dataset_name);
data_obj = load(strcat('Data/', dataset_name));
A = data_obj.A;
B = data_obj.B;
data = data_obj.data;  

disp(A);
disp(B);

%A = A(1:8, 1:8); % keep only the part of A corresponding to the omega dynamic
%B = B(1:8, 1); % keep only the part of B corresponding to the omega dynamic

%disp(A);
%disp(B);


figure(1);
histogram(data(4, :), 2);
title('Distribution of the mode q in the dataset', 'Interpreter', 'latex'); 

figure(2);
histogram(data(5, :), 100);
title('Distribution of the u in the dataset', 'Interpreter', 'latex'); 

% Recreate the DC motor object.
sys = DCMotorHybridSystemClass();

% Define the observation function y = h(x, t)
h = @(x, t) x(2);
obs_sys = ObservedHybridSystem(sys, 1, h);

n_z = size(A, 1); % dimension of the z dynamic
% Define the AugmentedSystem with the same z dynamic
aug_sys = AugmentedSystem(obs_sys, n_z, A, B);

% Extract z and x components from the dataset.
X = data(aug_sys.nx + 1 : aug_sys.nx + aug_sys.nz, :)';
Y = data(1 : aug_sys.nx, :)';

U_input = data(5:6, :)'; % extract u and u_dot from x 
U_pulsation = data(9, :)'; % extract the pulsation of the input sine wave from the dataset
X = cat(2, X, U_input, U_pulsation); % we keep u and u_dot as input of the predictor

State = Y(:, 1:3); % current, angular velocity and rotor angle
Friction_coef= Y(:, 7:8); % friction coefficients

Y = cat(2, State, Friction_coef); % we want to predict the current, angular velocity, rotor angle and friction coefficients from z, u, u_dot and pulsation

% Remove NaN rows if present.
valid = all(~isnan(X), 2) & all(~isnan(Y), 2);
fprintf( '%.0f nan over %.0f data points \n',sum(~valid), length(valid));

X = X(valid, :);
Y = Y(valid, :);

% Split into training and test sets.
cv = cvpartition(size(Y, 1), 'HoldOut', 0.3);
X_train = X(training(cv), :);
Y_train = Y(training(cv), :);
X_test = X(test(cv), :);
Y_test = Y(test(cv), :);

disp(size(X_train));
% Normalize the z data.
[X_train, mu, sigma] = zscore(X_train);
X_test = (X_test - mu) ./ sigma;

% Regression network from z to x.
layers = [
    featureInputLayer(11) % we also input u and u_dot to the predictor
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(100)
    tanhLayer
    fullyConnectedLayer(5)
    regressionLayer];

% Training options
options = trainingOptions('adam', ...
    'ValidationData',{X_test, Y_test}, ...
    'ValidationFrequency',30, ...
    'MaxEpochs', 100, ...
    'ValidationPatience', 300, ...
    'ObjectiveMetricName',"loss", ...
    'OutputNetwork', 'best-validation', ...
    'InitialLearnRate', 0.01, ...
    'L2Regularization', 1e-5, ...
    'Plots', 'training-progress');


mdl = trainNetwork(X_train, Y_train, layers, options);

Y_pred = predict(mdl, X_test);
rmse = sqrt(mean((Y_pred - Y_test).^2, 'all'));
fprintf('DC Motor inverse gluing RMSE: %.4f\n', rmse);

% Save the trained model.
today = string(datetime('today'));
model_filename = strcat('ObserverModels/dc-motor-predictor-sparse-', today);
save(model_filename, 'mdl', 'mu', 'sigma', 'A', 'B', 'dataset_name');
