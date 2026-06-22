%% Example: Bouncing Ball 
% In this example, we illustrate the Gluing methodology. We create
% a BouncingBall subject to friction using the HybridSystem class.
% We then generate and save a (x, z, labels) dataset using the AugmentedSystem class
% in view of training a model of the inverse gluing transformation giving x
% from z.

% Create a BouncingBall object.
addpath('utils', 'Examples/BouncingBall');
sys = BouncingBallSystemClass();
sys.mu = 2; % Additional velocity at each impact

% Create an observed system

% Define the observation function y = h(x, t)
h = @(x, t) (x(1));

% Create the associated BouncingBall object
obs_sys = ObservedHybridSystem(sys, 1, h);
nx = 25;
% Define the z dynamic :  z' = Az + Bh(x)
halfnz = 2;
nz = 2 * halfnz;
maxReal = 8;
maxImag = 4;
realParts = -maxReal * rand(halfnz, 1);
imagParts = maxImag * (rand(halfnz, 1) - 1/2) * 2;
poles = [realParts + 1i * imagParts; realParts - 1i * imagParts];
plot(real(poles), imag(poles), 'o')

A = [realParts(1), -imagParts(1);imagParts(1), realParts(1)];
for k = 2:halfnz
    A = [A zeros(2 * (k-1), 2);zeros(2, 2* (k-1)) [realParts(k), -imagParts(k);imagParts(k), realParts(k)]];
end

B = ones(nz,1);

aug_sys = AugmentedSystem(obs_sys, nz, A, B);

%% Generate a labeled dataset of (x,z) pair

nInit = 40;
% Random initial conditions sampled uniformly inside a specific rectangle
Init_conditions = aug_sys.generateRandomConditions([0, 5; -12, 12], nInit);

% Choose a time after which the z dynamic is in stationnary state
t_take = 5/min(abs(real(eig(A))));
% Generate the dataset from 400 initial conditions, with 200 points stored per trajectory, chosen between t_take and t_take + 15s, max_dt of ODE solver : 0.001s
data = aug_sys.generateData(Init_conditions, t_take, t_take + 15, 200, nInit, 0.001);

%% Save dataset
today = string(datetime("today"));
datas_filename = strcat('Data/raw-bouncing-ball-', today);
save(datas_filename, "data", "A", "B")  % save labelled dataset and the corresponding z dynamic used to generate it