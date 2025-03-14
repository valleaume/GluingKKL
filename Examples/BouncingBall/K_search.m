sys_ball = BouncingBallSubSystemClass();

sys_obs = BouncingBallObserver();
sys_obs.L_c = [0.1; 0.25];
sys_obs.L_d = 1*[0.1; 0.1];

sys_ball.mu = 0; % Additional velocity at each impact

sys_ball.lambda = 1;
sys_ball.f_air = 0;

sys_obs.mu = sys_ball.mu;
sys_obs.lambda = sys_ball.lambda;
sys_obs.f_air = sys_ball.f_air;


sys_obs.L_c = [lc_1; lc_2];
sys_obs.L_d = [ld_1; ld_2];
% BEWARE : only true for no air friction
F = [0, 1; 0, 0];
J = [1, 0; 0, -sys_ball.lambda];

H = [1, 0];
w = [1; 0];
tau = 15.6718 - 13.6045;
x = [0; -10.0995];

for ld_1 = -5:0.01:5
    for ld_2 = -5:0.01:5
        for lc_1 = 0:0.01:0.4
            for lc_2 = 0:0.01:0.4

                M_before = J - sys_obs.L_d*H - (J*sys_ball.flowMap(x, 0, 0, 0) - sys_ball.flowMap(sys_ball.jumpMap(x, 0, 0, 0), 0, 0, 0) )/x(2)*w';
                M_after = M_before - sys_obs.L_d*H*(sys_ball.flowMap(sys_ball.jumpMap(x, 0, 0, 0), 0, 0, 0) - sys_ball.flowMap(x, 0, 0, 0))/x(2)*w';

                if all(abs(eig(M_after*expm((F-sys_obs.L_c*H)*tau))) <= 1)
                    if all(abs(eig(M_before*expm((F-sys_obs.L_c*H)*tau))) <= 1)
                        disp(sys_obs.L_d, sys_obs.L_c)
                        break
                    end
                end
            end
        end
    end
end


