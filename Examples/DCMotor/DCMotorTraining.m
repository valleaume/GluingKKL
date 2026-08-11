%% DCMotorTraining.m
% Load the most recent DC motor dataset and train inverse KKL predictors.
%
% Steps:
%  1. Locate and load the most recent dataset saved by
%     `DCMotorDataGeneration.m`.
%  2. Reconstruct the augmented system dimensions (A,B) and extract
%     z-features and corresponding physical states/parameters.
%  3. Preprocess (NaN removal, normalization) and split data.
%  4. Train two neural regressors (parameters and states) and save models.

addpath('utils', 'Examples/DCMotor');
close all;

% Find the latest generated DC motor dataset (file name pattern
% 'Data/raw-dc-motor-*.mat'). We expect `data` plus saved `A` and `B`.
files = dir('Data/raw-dc-motor-*.mat');
assert(~isempty(files), 'No raw DC motor dataset found in Data/');
[~, idx] = max([files.datenum]);
dataset_name = files(idx).name;
disp(['Loading dataset: ', dataset_name]);
data_obj = load(fullfile('Data', dataset_name));
% Unpack saved matrices and the data matrix
A = data_obj.A;
B = data_obj.B;
data = data_obj.data;

disp('Loaded A:'); disp(A);
disp('Loaded B:'); disp(B);

%A = A(1:8, 1:8); % keep only the part of A corresponding to the omega dynamic
%B = B(1:8, 1); % keep only the part of B corresponding to the omega dynamic

%disp(A);
%disp(B);


figure(1);
subplot(1, 3, 1);
histogram(data(4, :), 2);
title('Distribution of the mode q  ', 'Interpreter', 'latex'); 

subplot(1, 3, 2);
histogram(data(5, :), 100);
title('Distribution of the u  ', 'Interpreter', 'latex'); 



% Recreate the DC motor model and observed-system wrapper. The observed
% map `h` must match the observation used during dataset generation.
sys = DCMotorHybridSystemClass();
% Observe angular velocity and current (ordering must match original script)
h = @(x, t) [x(2), x(1)];
ny = 2;
obs_sys = ObservedHybridSystem(sys, ny, h);

% Dimension of the z-dynamics (from saved A)
n_z = size(A, 1);
% Create AugmentedSystem to reuse helper methods/fields (nx, nz, etc.)
aug_sys = AugmentedSystem(obs_sys, n_z, A, B);

% Extract z-features (rows after the state in `data`) and the measured
% state vector `Y` (first `nx` rows). The `data` matrix layout is [x; z].
X = data(aug_sys.nx + 1 : aug_sys.nx + aug_sys.nz, :)';
Y = data(1 : aug_sys.nx, :)';

% Additional inputs appended to the predictor input: `u` and `u_dot` and
% the input pulsation parameter. Adjust indices if `data` layout changes.
U_input = data(5:6, :)';        % columns corresponding to u and u_dot
U_pulsation = data(9, :)';      % pulsation of the input sine wave
X = cat(2, X, U_input, U_pulsation);

% Define predictor targets: here we predict (current, angular velocity,
% rotor angle) and friction coefficients (F_s, F_d). Ordering must match
% the downstream uses of the predictors.
State = Y(:, 1:3);            % current, angular velocity, rotor angle
Friction_coef= Y(:, 7:8);    % friction coefficients (F_s, F_d)
Y = cat(2, State, Friction_coef);

% Remove rows containing NaNs in either inputs or targets
valid = all(~isnan(X), 2) & all(~isnan(Y), 2);
fprintf('%.0f NaN rows removed over %.0f data points\n', sum(~valid), length(valid));
X = X(valid, :);
Y = Y(valid, :);

% subplot(1, 3, 3);
% histogram(U_input(valid(1:n_plot), 1), 100);  
% title('Distribution of the input  , after Nan', 'Interpreter', 'latex');

% figure(2);
% subplot(1, 2, 1);
% histogram(data(1, :), 100);
% title('Distribution of the current $i$  ', 'Interpreter', 'latex'); 

% subplot(1, 2, 2);
% histogram(Y(:, 1), 100);  
% title('Distribution of the current $i$  , after NaN removal', 'Interpreter', 'latex'); 

% figure(3);
% subplot(1, 2, 1);
% histogram(data(2, :), 100);
% title('Distribution of the angular velocity $\omega$  ', 'Interpreter', 'latex'); 

% subplot(1, 2, 2);
% histogram(Y(:, 2), 100);  
% title('Distribution of the angular velocity $\omega$  , after NaN removal', 'Interpreter', 'latex'); 

% figure(4);
% subplot(1, 2, 1);
% histogram(data(3, :), 100);
% title('Distribution of the rotor angle $\theta$  ', 'Interpreter', 'latex'); 

% subplot(1, 2, 2);
% histogram(Y(:, 3), 100);  
% title('Distribution of the rotor angle $\theta$  , after NaN removal', 'Interpreter', 'latex'); 

% Split into training and test sets (30% holdout)
cv = cvpartition(size(Y, 1), 'HoldOut', 0.3);
X_train = X(training(cv), :);
Y_train = Y(training(cv), :);
X_test = X(test(cv), :);
Y_test = Y(test(cv), :);

disp('Training set size:'); disp(size(X_train));
% Normalize input features using training set statistics
[X_train, mu, sigma] = zscore(X_train);
X_test = (X_test - mu) ./ sigma;

% Define a regression network mapping [z; u; u_dot; pulsation] -> target
layers = [
    featureInputLayer(n_z+3) % input: z-dimension plus u, u_dot and pulsation
    fullyConnectedLayer(128)
    tanhLayer
    fullyConnectedLayer(64)
    tanhLayer
    fullyConnectedLayer(64)
    tanhLayer
    fullyConnectedLayer(2)   % output: two parameters (F_s, F_d)
    regressionLayer];

% Training options
% Training options for parameter predictor (uses validation on parameter
% columns 4:5 of `Y_test` which correspond to friction coefficients)
options = trainingOptions('adam', ...
    'ValidationData',{X_test, Y_test(:, 4:5)}, ...
    'ValidationFrequency',30, ...
    'MaxEpochs', 100, ...
    'ValidationPatience', 300, ...
    'ObjectiveMetricName',"loss", ...
    'OutputNetwork', 'best-validation', ...
    'InitialLearnRate', 0.01, ...
    'L2Regularization', 1e-5, ...
    'Plots', 'training-progress');

disp("start training")
mdl_p = trainNetwork(X_train, Y_train(:, 4:5), layers, options);

% Evaluate parameter predictor on test set
Y_pred = predict(mdl_p, X_test);
rmse = sqrt(mean((Y_pred - Y_test(:, 4:5)).^2, 'all'));
fprintf('DC Motor inverse KKL, parameter estimation RMSE: %.4f\n', rmse);

% Save the trained parameter model and preprocessing stats
today = string(datetime('today'));
model_filename = strcat('ObserverModels/dc-motor-predictor-parameter-', today);
save(model_filename, 'mdl_p', 'mu', 'sigma', 'A', 'B', 'dataset_name');

% Regression network mapping to system state (current, angular velocity,
% rotor angle). Architecture similar to parameter predictor but outputs 3 values.
layers = [
    featureInputLayer(n_z+3)
    fullyConnectedLayer(128)
    tanhLayer
    fullyConnectedLayer(64)
    tanhLayer
    fullyConnectedLayer(64)
    tanhLayer
    fullyConnectedLayer(3)   % output: current, angular velocity, rotor angle
    regressionLayer];

% Training options
options = trainingOptions('adam', ...
    'ValidationData',{X_test, Y_test(:, 1:3)}, ...
    'ValidationFrequency',30, ...
    'MaxEpochs', 100, ...
    'ValidationPatience', 300, ...
    'ObjectiveMetricName',"loss", ...
    'OutputNetwork', 'best-validation', ...
    'InitialLearnRate', 0.01, ...
    'L2Regularization', 1e-5, ...
    'Plots', 'training-progress');

% Train state predictor and evaluate
mdl_s = trainNetwork(X_train, Y_train(:, 1:3), layers, options);
Y_pred = predict(mdl_s, X_test);
rmse = sqrt(mean((Y_pred - Y_test(:, 1:3)).^2, 'all'));
fprintf('DC Motor inverse KKL, state estimation RMSE: %.4f\n', rmse);

% Save the trained state model
model_filename = strcat('ObserverModels/dc-motor-predictor-state-', today);
save(model_filename, 'mdl_s', 'mu', 'sigma', 'A', 'B', 'dataset_name');
