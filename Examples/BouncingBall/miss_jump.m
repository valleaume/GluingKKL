addpath('utils', 'Examples/BouncingBall', "Examples/StickSlip");
close all; % close all previously opened figures

% ReCreate the BouncingBall object.
sys = ObserverBouncingBallSystemClass();
sys.mu = 2; % Additional velocity at each impact

% Initial condition
X0_s = [[5.; 2.; 5.01; 2.01], 1*[5.; 2.; 5.01; 2.01]];  % system initial condition %[4.05; 2.; 5; 2.05]]
sys.l1 = 10;
sys.l2 = 25;
sys.k1 = 3;

% Time spans
tspan = [0, 10];
jspan = [0, 245];

% Specify solver options.
config = HybridSolverConfig('AbsTol', 1e-3, 'RelTol', 1e-7, 'MaxStep', 0.01 );
X0 = X0_s(:,2);
% Compute solution of the cascade system - observer
sol_test = sys.solve(X0, tspan, jspan, config);
figure(1)
plot(sol_test.x(:,1), sol_test.x(:,2));
hold on;
plot(sol_test.x(:,3), sol_test.x(:,4));

figure(2)
plot(sol_test.t, sol_test.x(:,1));
hold on;
plot(sol_test.t, sol_test.x(:,3));

figure(3)
plot(sol_test.t, sol_test.x(:,2));
hold on;
plot(sol_test.t, sol_test.x(:,4));



