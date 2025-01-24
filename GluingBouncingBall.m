%% Example: Bouncing Ball using HybridSystem
% In this example, we illustrate the Gluing methodology. We create and solve
% a BouncingBall suject to friction using the HybridSystem class.
% We then generate a dataset it using the AugmentedSystem class. Using the
% splitting methodology we train 2 neural networks and a classifier to
% learn T_inv. We then illustrate the observer.
% For a full description of this example, find the following page in the 
% HyEQ Toolbox Help: Hybrid Equations MATLAB Library > Creating and Simulating Hybrid Systems

%% Create a BouncingBall object.

sys = BouncingBallFriction();
sys.mu = 1; % Additional velocity at each impact

%% Create an observed system

% Define the observation function y = h(x)
h = @(x, t) (x(1));

% Create the associated BouncingBall object
obs_sys = ObservedHybridSystem(sys, 1, h);

% Define the AugmentedSystem
A = diag([-10,-20,-30]);
B = [1; 1; 1];
aug_sys = AugmentedSystem(obs_sys, 3, A, B);


%% Compute a solution

% Initial condition for the initial system
x0 =  [4; 0];

% Time spans
tspan = [0, 100];
jspan = [0, 15];

% Specify solver options.
config = HybridSolverConfig('AbsTol', 1e-3, 'RelTol', 1e-7);

% Compute solutions
sol = sys.solve(x0, tspan, jspan, config);

%% Plot the solution

figure(1)
clf
hpb = HybridPlotBuilder();
hpb.title('Ball')...
    .subplots('on')...
    .legend()...
    .plotFlows(sol)

%% Compute an augmented solution

% Initial condition for augmented system
X0 =  [4; 1; 0; 0; 0];

% Time spans
tspan = [0, 16];
jspan = [0, 20];

% Specify solver options.
config = HybridSolverConfig('AbsTol', 1e-3, 'RelTol', 1e-7);

% Compute solutions
sol_aug = aug_sys.solve(X0, tspan, jspan, config);


%% Plot Augmented solution
figure(2)
clf
hpb = HybridPlotBuilder();
hpb.title('Ball')...
    .subplots('on')...
    .legend()...
    .plotFlows(sol_aug)

%% Generate a labeled dataset of (x,z) pair

% Random initial conditions sampled uniformly inside a specific rectangle
Init_conditions = aug_sys.generateRandomConditions([0, 5; -12, 12], 400);

% Generate the dataset
data = aug_sys.generateData(Init_conditions, 0.7, 20, 200, 400);
%data_3 = aug_sys.generateData(Init_conditions, 0.7, 20, 200, 400);

%% Save dataset
today = string(datetime("today"));
datas_filename = strcat('Data/raw-bouncing-ball-', today);
data_3 = data; %save dataas "data_3" for compatibility purposes
save(datas_filename, "data_3", "A", "B")

%% Debug Labelling
% baseline = aug_sys.generateDataTest(Init_conditions, 0.7, 20, 200, 1);
% test_a = aug_sys.generateData(Init_conditions, 0.7, 20, 200, 1);
% 
% disp(size(baseline{2}.x))
% figure(8)
% clf
% scatter(baseline{2}.t(baseline{1}==1), baseline{2}.x(baseline{1}==1, 1), 4, 'red')
% hold on
% scatter(baseline{2}.t(baseline{1}==0), baseline{2}.x(baseline{1}==0, 1), 40, 'blue')
% 
% scatter(test_a{2}.t(test_a{1}==1), test_a{2}.x(test_a{1}==1, 1), 15, 'red')
% 
% scatter(test_a{2}.t(test_a{1}==0), test_a{2}.x(test_a{1}==0, 1), 55, 'blue')

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
nTrees = 100; % nombre d'arbres dans la forêt
maxTreeDepth = 30; % profondeur maximale de l'arbre
minSampleNode = 1; % échantillon minimum par nœud
%polynomialDegree = 2; % degré polynomial (notez que ceci n'est pas directement applicable dans TreeBagger)
minSampleSplit = 10; % échantillon minimum pour la division des nœuds

% Créer des options pour l'entraînement
opts = statset('UseParallel', true);

% Entraîner la forêt
randomForest = TreeBagger(nTrees, XTrain, YTrain, 'Method', 'classification')%, ...
 %   'MinLeafSize', minSampleNode);
%,     'MaxNumSplits', maxTreeDepth, ... ...
 %   'Options', opts, ...
  %  'Surrogate', 'on');
%'MinParentSize', minSampleSplit);
svmModel = fitcsvm(XTrain, YTrain, 'KernelFunction', 'rbf', 'Standardize', true)


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
mask_a = reshape(data(6, :)==1,1,[]);
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

mask_b = reshape(data(6, :) == 0, 1, []);


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
directory = 'ObserverModels/';
models_name = strcat(directory, 'bouncing-ball-predictor-', today,'.mat');
save(models_name, 'mdl_b', 'mdl_b', "mu_b", "sigma_b", "mdl_a", "mu_a", "sigma_a", "svmModel", "randomForest");

%% Test performances of the fully fledged observer

%% Generate a ground truth x signal and its corresponding z signal

% Initial condition
X1 =  [5; 2; 0; 0; 0];

% Time spans
tspan = [0, 26];
jspan = [0, 28];

% Specify solver options.
config = HybridSolverConfig('AbsTol', 1e-3, 'RelTol', 1e-7);

% Compute solutions
sol_test = aug_sys.solve(X1, tspan, jspan, config);

y = sol_test.x(:,1);
z = sol_test.x(:,3:5);


%% Reconstruct the observer result

models = load(models_name);
T_inv = Predictor(models);
x_pred = T_inv.predict(z);

%% Plot observer result and ground truth
figure(6)
clf
plot(sol_test.x(:,1), sol_test.x(:,2))
plot(x_pred(:,1), x_pred(:,2))
plot(sol_test.t(:), sol_test.x(:,2))
hold on
plot(sol_test.t(:), x_pred(:,2))
hold off 
plot(sol_test.t(:), sol_test.x(:,1))
hold on
plot(sol_test.t(:), x_pred(:,1))
hold off


%% Noisy data

sig = 0.01;

% Define the observation function y = h(x) + noise
h_noise = @(x) (x(1) + sig*randn);

% Create the associated BouncingBall object
obs_sys_noise = ObservedHybridSystem(sys, 1, h_noise);

aug_sys_noise = AugmentedSystem(obs_sys_noise, 3, A, B);

sol_test_noise = aug_sys_noise.solve(X1, tspan, jspan, config);


z_noise = sol_test_noise.x(:,3:5);

%% Reconstruct the observer result

x_pred_noise = T_inv.predict(z_noise);

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


