%% Example:StickSlip using HybridSystem
% In this example, we illustrate the Gluing methodology. We create and solve a StickSlip suject to friction using the HybridSystem class.
% We then generate a dataset it using the AugmentedSystem class. Using the
% splitting methodology we train 2 neural networks and a classifier to
% learn T_inv. We then illustrate the observer.
% For a full description of this example, find the following page in the HyEQ Toolbox Help: Hybrid Equations MATLAB Library > Creating and Simulating Hybrid Systems

%% Create a StickSlip object.

sys = StickSlip();

%% Load pre-existing models and corresponding datas

obj = load("ObserverModels/StickSlip-adaptedFrequenciesPredictors.mat");
svmModel = obj.svmModel;
mdl_b = obj.mdl_b;
mdl_a = obj.mdl_a;
sigma_a = obj.sigma_a;
sigma_b = obj.sigma_b;
mu_a = obj.mu_a;
mu_b = obj.mu_b;
randomForest = obj.randomForest;  

datas = load("Data/raw_data-adaptedFrequencies.mat");
A = datas.A;
B = datas.B;

assert(size(A,1) == size(B,1), 'A and B are not consistent')

%% Create an observed system

% Define the observation function y = h(x)
h = @(x, t) (x(1));

% Create the associatedStickSlip object
obs_sys = ObservedHybridSystem(sys, 1, h);

% Define the AugmentedSystem

aug_sys = AugmentedSystem(obs_sys, size(A, 1), A, B);


%% Test performances of the fully fledged observer

%% Generate a ground truth x signal and its corresponding z signal

% Initial condition
X1 =  [0; sys.v_t; 0.45; 0.3; -1; 0; 0; 0; 0; 0; 0];

% Time spans
tspan = [0, 260];
jspan = [0, 280];

% Specify solver options.
config = HybridSolverConfig('AbsTol', 1e-3, 'RelTol', 1e-7);

% Compute solutions
sol_test = aug_sys.solve(X1, tspan, jspan, config);

y = sol_test.x(:,1);
z = sol_test.x(:, aug_sys.nx + 1 : aug_sys.nx + aug_sys.nz);


%% Reconstruct the observer result

% Classify z points
after_jumps_label = str2double(predict(randomForest, z));

% Calculate results of both networkd
x_pred_before_jump = predict(mdl_b, (z - mu_b) ./ sigma_b);
x_pred_after_jump = predict(mdl_a, (z - mu_a) ./ sigma_a);

% Assign the correct result
x_pred = x_pred_before_jump;
x_pred(after_jumps_label==1,:) = x_pred_after_jump(after_jumps_label==1,:);

%% Plot observer result and ground truth
figure(6)
clf
plot(sol_test.x(:,1), sol_test.x(:,2))
title('True PhasePlot')
plot(x_pred(:,1), x_pred(:,2))
title('Observed PhasePlot')
plot(sol_test.t(:), sol_test.x(:,2))

hold on
plot(sol_test.t(:), x_pred(:,2))
title('velocity')
hold off 
plot(sol_test.t(:), sol_test.x(:,1))
title('position')
hold on
plot(sol_test.t(:), x_pred(:,1))
hold off
plot(sol_test.t(:), sol_test.x(:,3))
title('\mu_s')
hold on
plot(sol_test.t(:), x_pred(:,3))
hold off
plot(sol_test.t(:), sol_test.x(:,4))
title('\mu_d')
hold on
plot(sol_test.t(:), x_pred(:,4))
hold off
plot(sol_test.t(:), sol_test.x(:,5))
title('q')
hold on
plot(sol_test.t(:), x_pred(:,5))


%% Noisy data

sig = 0.01;

% Define the observation function y = h(x) + noise
h_noise = @(x,t) (x(1) + sig*sin(t));

% Create the associated StickSlip object
obs_sys_noise = ObservedHybridSystem(sys, 1, h_noise);

aug_sys_noise = AugmentedSystem(obs_sys_noise, aug_sys.nz, A, B);

sol_test_noise = aug_sys_noise.solve(X1, tspan, jspan, config);


z_noise = sol_test_noise.x(:, aug_sys_noise.nx + 1:aug_sys_noise.state_dimension);

%% Reconstruct the observer result

% Classify z points
after_jumps_label_noise = str2double(predict(randomForest, z_noise));

% Calculate results of both networkd
x_pred_before_jump = predict(mdl_b, (z_noise - mu_b) ./ sigma_b);
x_pred_after_jump = predict(mdl_a, (z_noise - mu_a) ./ sigma_a);

% Assign the correct result
x_pred_noise = x_pred_before_jump;
x_pred_noise(after_jumps_label_noise == 1,:) = x_pred_after_jump(after_jumps_label_noise == 1,:);

%% Plot observer result and ground truth
figure(7)
clf
plot(sol_test_noise.x(:,1), sol_test_noise.x(:,2))
plot(x_pred_noise(:,1), x_pred_noise(:,2))
plot(sol_test_noise.t(:), sol_test_noise.x(:,2))
hold on
plot(sol_test_noise.t(:), x_pred_noise(:,2))
hold off 
plot(sol_test_noise.t(:), sol_test_noise.x(:,1))
hold on
plot(sol_test_noise.t(:), x_pred_noise(:,1))
hold off


%% Import a labeled dataset of (x,z) pair

%datas = load("Data\raw_data-slow.mat"); already imported
%data_3 = aug_sys.generateData(Init_conditions, 0.7, 20, 200, 400);
data_3 = datas.data_3;


%% Plot the 2 classes of points

% Plot in the x space
figure(3)
clf
scatter(data_3(1, data_3(aug_sys.state_dimension + 1,:)==1), data_3(2, data_3(aug_sys.state_dimension + 1,:)==1), 8, 'r')
hold on
scatter(data_3(1, data_3(aug_sys.state_dimension + 1,:)==0), data_3(2, data_3(aug_sys.state_dimension + 1,:)==0), 8, 'b')
xlabel('x_1')
ylabel('x_2')
%scatter(Init_conditions(1,:), Init_conditions(2,:), 4)
legend('After Jump', 'Before Jump', 'Init Conditions' )

% Plot in the x space
figure(4)
clf
scatter3(data_3(1, data_3(aug_sys.state_dimension + 1,:)==1), data_3(2, data_3(aug_sys.state_dimension + 1,:)==1), data_3(3, data_3(aug_sys.state_dimension + 1,:)==1), 8, 'r')
hold on
scatter3(data_3(1, data_3(aug_sys.state_dimension + 1,:)==0), data_3(2, data_3(aug_sys.state_dimension + 1,:)==0), data_3(3, data_3(aug_sys.state_dimension + 1,:)==0), 8, 'b')
%scatter3(data_3(3, isnan(data_3(6,:))), data_3(4, isnan(data_3(6,:))), data_3(5, isnan(data_3(6,:))), 5, 'black')
xlabel('z_1')
ylabel('z_2')
zlabel('z_3')
legend('After Jump', 'Before Jump', 'Nan' )

%% Remove Nan
mask = reshape(~isnan(data_3(aug_sys.state_dimension + 1, :)), 1, []);
disp(sum(~mask));
X = data_3(aug_sys.nx + 1:aug_sys.nx+aug_sys.nz, mask);
Y = data_3(aug_sys.state_dimension + 1, mask);

%% Test the classifier
% 2 models are tested : svm with radial kernels and random forest

%% Test and train split
cv_par_t = cvpartition(Y, 'HoldOut', 0.3);
disp(cv_par_t);

%XTrain = X(:, training(cv_par_t))';
%YTrain = Y(training(cv_par_t))';
XTest = X(:, test(cv_par_t))';
YTest = Y(test(cv_par_t))';

%XTrain = XTrain(data(1, training(cv_par_t))>0.1, :);
%YTrain = YTrain(data(1, training(cv_par_t))>0.1, :);





%% Test
% Faire des prédictions sur l'ensemble de test
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
X_x = data_3(1:2, mask);
X_x = X_x(:, test(cv_par_t));
disp(size(X_x));
scatter(X_x(1, false_flag), X_x(2, false_flag))
xlabel('x_1')
ylabel('x_2')
title('Missclassified points')

%% Test the neural networks

mask_a = reshape(data_3(aug_sys.state_dimension + 3, :) == 1, 1, []);
X_a = data_3(aug_sys.nx + 1:aug_sys.nx + aug_sys.nz, mask_a)';
Y_a = data_3(1:aug_sys.nx, mask_a)';

cv = cvpartition(size(Y_a, 1), 'HoldOut', 0.3);
XTrain_a = X_a(training(cv), :);
YTrain_a = Y_a(training(cv), :);
XTest_a = X_a(test(cv), :);
YTest_a = Y_a(test(cv), :);

% Normaliser les données
XTrain_a = (XTrain_a - mu_a) ./ sigma_a;
XTest_a = (XTest_a - mu_a) ./ sigma_a;


mask_b = reshape(data_3(aug_sys.state_dimension + 2, :) == 1, 1, []);
X_b = data_3(aug_sys.nx + 1: aug_sys.nx + aug_sys.nz, mask_b)';
Y_b = data_3(1:aug_sys.nx, mask_b)';

% Fractionner les données en ensemble d'entraînement et de test
cv = cvpartition(size(Y_b, 1), 'HoldOut', 0.3);
XTrain_b = X_b(training(cv), :);
YTrain_b = Y_b(training(cv), :);
XTest_b = X_b(test(cv), :);
YTest_b = Y_b(test(cv), :);

% Normaliser les données
XTrain_b = (XTrain_b - mu_b) ./ sigma_b;
XTest_b = (XTest_b - mu_b) ./ sigma_b;

YPred_b = predict(mdl_b, XTest_b);

% Évaluer la précision (utiliser une métrique de régression, comme la RMSE)
rmse = sqrt(mean((YPred_b - YTest_b).^2, 'all'));
fprintf('RMSE before jumps : %.4f\n', rmse);

YPred_a = predict(mdl_a, XTest_a);

% Évaluer la précision (utiliser une métrique de régression, comme la RMSE)
rmse = sqrt(mean((YPred_a - YTest_a).^2, 'all'));
fprintf('RMSE after jumps : %.4f\n', rmse);
