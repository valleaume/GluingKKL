classdef AugmentedSystem < HybridSystem  %cannot ihnerit from Observable sys?
    properties(SetAccess = immutable)
        x_indices;
        z_indices;
        A;
        B;
        nx;
        nz;
        n;
        baseSystem;
    end
    methods
        function this = AugmentedSystem(BaseSystem, nz, A, B)
            nx = BaseSystem.state_dimension;
            state_dim = nz + nx;
            this = this@HybridSystem(state_dim);

            this.x_indices = 1:nx;
            this.z_indices = nx + (1:nz);
            
            this.nx = nx;
            this.nz = nz;
            this.A = A;
            this.B = B;

            this.baseSystem = BaseSystem;
        end

        function Xdot = flowMap(this, X, t, j)
            x = X(this.x_indices);
            z = X(this.z_indices);
            xdot = this.baseSystem.flowMap(x);
            zdot = this.A*z + (this.baseSystem.h(x,t)*this.B')';
            Xdot = [xdot; zdot];
        end

        function Xplus = jumpMap(this, X, t, j)
            x = X(this.x_indices);
            z = X(this.z_indices);
            xplus = this.baseSystem.jumpMap(x);
            zplus = z;
            Xplus = [xplus; zplus];
        end

        function inC = flowSetIndicator(this, X, t, j)
            % Extract the state components
            x = X(this.x_indices);
            inC = this.baseSystem.flowSetIndicator(x);
        end

        function inD = jumpSetIndicator(this, X, t, j)
            % Extract the state components
            x = X(this.x_indices);
            inD = this.baseSystem.jumpSetIndicator(x);
        end

        function InitConditions = generateUniformConditions(this, bounds, n)
            X0_bound = cell(this.nx);
            for i = 1:this.nx
                X0_bound(i) = linspace(bounds(i,1), bounds(i,2), n);
            end
            output = ndgrid(X0_bound{:});
            InitConditions = zeros(n^this.nx, this.nx);
            for  i = 1:this.nx
                InitConditions(:, i) = output;
            end
        end

        function InitConditions = generateRandomConditions(this, bounds, n_points)
            seed = RandStream('mlfg6331_64');
            InitConditions = rand(seed, this.nx, n_points) .* (bounds(:, 2) - bounds(:, 1)) + bounds(:, 1);
        end

        function DataSet = generateData(this, InitConditions, T_take, T_max, points_per_run, n_run, max_dt_step, train_margin)
            %GENERATE DATA Method that generate a dataset of points and 
            %label them. 
            %   The points are randomly sampled alongside trajectories
            %   of the augmented system initialized thanks to the input
            %   array. They can be labelled as before (label = 0) or after
            %   (label = 1) their closest jump. Sampling is started after
            %   T_take time so that the transitory period does not affect
            %   much the points. 
            %   INPUTS : 
            %       - InitConditions : (M, n_x + n_z) array, with M >= n_run
            %       - T_take : Time after which the transitory is finished
            %       - T_max : End time for simulations
            %       - points_per_run : Number of points randomly sampled each run
            %       - n_run : Number of run
            %   OUTPUT :
            %       DataSet : (n_run, points_per_run, n_x + n_z + 3) array

            
            assert(T_take < T_max, "T_take is greater than T_max")
            assert(size(InitConditions, 2) >= n_run, "Not enough initial condition for required number of runs")
            assert(T_take > 5/abs(eigs(this.A, 1, 'smallestabs')), "T_take not big enough compared to z dynamic")
            if ~(exist('train_margin', 'var'))
                train_margin = 0.1;
            end

            if ~(exist('max_dt_step', 'var'))
                max_dt_step = 0.01;
            end

            config = HybridSolverConfig('silent', 'AbsTol', 1e-3, 'RelTol', 1e-7, 'MaxStep', max_dt_step);

            J_max = 1000; 
            J_init = 0;
            tspan = [0, T_max]; 
            jspan = [J_init, J_max]; % Jump Span, make it very large to set the stopping condition to be a certain time, not a certain number of jumps (except if Zeno)
            
            
            DataSet = nan(n_run, points_per_run, this.state_dimension + 3);
            AugmentedInitConditions = cat(1, InitConditions(:, 1:n_run), zeros(this.nz, n_run));
            seed = RandStream('mlfg6331_64');
            
            for i = 1:n_run
                AugmentedIc = AugmentedInitConditions(:, i); 
                sol = this.solve(AugmentedIc, tspan, jspan, config);
            
                data_x = sol.x;  % Data of (x,z)
                data_t = sol.t;  % Data of (t)
                data_j = sol.j;  % Data of j

                data_t_ind = size(data_t);  % Indices span
                last_index = data_t_ind(1);
                tmin_ind = find(data_t > T_take, 1);  % Min index of time such that t > T_take
                tmax_ind = last_index;  % Max index of time

                if data_j(tmax_ind) - data_j(tmin_ind) < 2  % ensure there is at least 1 complete jump after T_take (otherwise there is no interest in splitting)
                    continue;
                end

                % Redefine tmin_ind to be exactly the first point of the first complete jump after T_take
                for k1 = tmin_ind : tmax_ind-1
                    if data_j(k1+1) - data_j(k1) ~= 0
                        tmin_ind = k1+1;
                        break
                    end
                end
                
                % Redefine tmax_ind to be exactly the last point of the last complete jump
                %disp(sol.jump_count);
                tmax_ind = find(data_t == sol.jump_times(sol.jump_count), 1, 'last') - 1;
                %disp({data_j(tmax_ind), data_j(tmax_ind-1)});
                len_t = tmax_ind - tmin_ind + 1;

                j_int = data_j(tmin_ind);  % j value of the first complete jump after T_take
                %j_end = data_j(tmax_ind);  % j value of the last complete jump  
                after_jumps_label = nan(last_index, 1);
                to_train_after = zeros(last_index, 1);
                to_train_before = zeros(last_index, 1);
                
                current_J = j_int;
                middle_time = sol.jump_times(current_J - J_init) + (sol.jump_times(current_J - J_init + 1) - sol.jump_times(current_J - J_init))/2;
                middle_time_plus_margin = sol.jump_times(current_J - J_init) + (sol.jump_times(current_J - J_init + 1) - sol.jump_times(current_J - J_init))*(1/2 + train_margin);
                middle_time_minus_margin = sol.jump_times(current_J - J_init) + (sol.jump_times(current_J - J_init + 1) - sol.jump_times(current_J - J_init))*(1/2 - train_margin);
                DataSet_index = randsample(seed, tmin_ind:tmax_ind, points_per_run, points_per_run>len_t);
                DataSet_index = sort(DataSet_index);
                tab = tmin_ind:tmax_ind;
                for ind_old = 1:points_per_run
                    ind = DataSet_index(ind_old);
                    if data_j(ind)>current_J
                          current_J = data_j(ind);
                          %disp(current_J);
                          middle_time = sol.jump_times(current_J - J_init) + (sol.jump_times(current_J - J_init + 1) - sol.jump_times(current_J - J_init))/2;
                          middle_time_plus_margin = sol.jump_times(current_J - J_init) + (sol.jump_times(current_J - J_init + 1) - sol.jump_times(current_J - J_init))*(1/2 + train_margin);
                          middle_time_minus_margin = sol.jump_times(current_J - J_init) + (sol.jump_times(current_J - J_init + 1) - sol.jump_times(current_J - J_init))*(1/2 - train_margin);
              
                          if middle_time == data_t(ind)
                              % If there is no flow, cannot label the data
                              disp('hum')
                              continue
                          end
                    end

                    if data_t(ind)<middle_time
                        after_jumps_label(ind) = 1;
                    else
                        after_jumps_label(ind) = 0;
                    end

                    if data_t(ind)<=middle_time_plus_margin
                        to_train_after(ind) = 1;
                    end

                    if data_t(ind)>=middle_time_minus_margin
                        to_train_before(ind) = 1;
                    end 
                end
                
                
                DataSet(i,:,:) = cat(2, data_x(DataSet_index,:), after_jumps_label(DataSet_index), to_train_before(DataSet_index), to_train_after(DataSet_index));
                %disp(DataSet_index);
                % disp(after_jumps_label);
                % disp(after_jumps_label(DataSet_index));
                
            end
            DataSet = reshape(permute(DataSet,[3, 1, 2]), this.state_dimension + 3, []);
            %DataSet = {after_jumps_label, sol};% debug
        end
       

            function DataSet = generateDataTest(this, InitConditions, T_take, T_max, points_per_run, n_run)
                %GENERATE DATA Method that generate a dataset of points and 
                %label them. 
                %   The points are randomly sampled alongside trajectories
                %   of the augmented system initialized thanks to the input
                %   array. They can be labelled as before (label = 0) or after
                %   (label = 1) their closest jump. Sampling is started after
                %   T_take time so that the transitory period does not affect
                %   much the points. 
                %   INPUTS : 
                %       - InitConditions : (M, n_x + n_z) array, with M >= n_run
                %       - T_take : Time after which the transitory is finished
                %       - T_max : End time for simulations
                %       - points_per_run : Number of points randomly sampled each run
                %       - n_run : Number of run
                %   OUTPUT :
                %       DataSet : (n_run, points_per_run, n_x + n_z +1) array
    
                
                assert(T_take < T_max, "T_take is greater than T_max")
                assert(size(InitConditions, 2) >= n_run, "Not enough initial condition for required number of runs")
                assert(T_take > 5/abs(eigs(this.A, 1, 'smallestabs')), "T_take not big enough compared to z dynamic")
    
                config = HybridSolverConfig('silent', 'AbsTol', 1e-3, 'RelTol', 1e-7);
    
                J_max = 1000; 
                J_init = 1;
                tspan = [0, T_max]; 
                jspan = [J_init, J_max]; % Jump Span, make it very large to set the stopping condition to be a certain time, not a certain number of jumps (except if Zeno)
                
                DataSet = nan(n_run, points_per_run, this.state_dimension + 1);
                AugmentedInitConditions = cat(1, InitConditions(:, 1:n_run), zeros(this.nz, n_run));
                seed = RandStream('mlfg6331_64');
    
                for i = 1:n_run
                    AugmentedIc = AugmentedInitConditions(:, i); 
                    sol = this.solve(AugmentedIc, tspan, jspan, config);
    
                    data_x = sol.x;  % Data of (x,z)
                    data_t = sol.t;  % Data of (t)
                    data_j = sol.j;  % Data of j
    
                    data_t_ind = size(data_t);  % Indices span
                    last_index = data_t_ind(1);
                    tmin_ind = find(data_t > T_take, 1);  % Min index of time such that t > T_take
                    tmax_ind = last_index;  % Max index of time
    
                    if data_j(tmax_ind) - data_j(tmin_ind) < 2  % ensure there is at least 1 complete jump after T_take (otherwise there is no interest in splitting)
                        continue;
                    end
    
                    % Redefine tmin_ind to be exactly the first point of the first complete jump after T_take
                    for k1 = tmin_ind : tmax_ind-1
                        if data_j(k1+1) - data_j(k1) ~= 0
                            tmin_ind = k1+1;
                            break
                        end
                    end
                    
                    % Redefine tmax_ind to be exactly the last point of the last complete jump
                    tmax_ind = find(data_t == sol.jump_times(sol.jump_count), 1, 'last') - 1;
    
                    len_t = tmax_ind - tmin_ind + 1;
    
                    j_int = data_j(tmin_ind);  % j value of the first complete jump after T_take
                    j_end = data_j(tmax_ind);  % j value of the last complete jump  
                    after_jumps_label = nan(last_index, 1);
                    
                    for j = j_int:j_end
                        start_jump_time = sol.jump_times(j - J_init);
                        end_jump_time = sol.jump_times(j - J_init + 1);
                        middle_time = start_jump_time + (end_jump_time - start_jump_time)/2;
                        after_jumps_label((data_t >= start_jump_time) & (data_t < middle_time) & (data_j == j)) = 1;
                        after_jumps_label((data_t <= end_jump_time) & (data_t >= middle_time) & (data_j == j)) = 0;
                    end
                    DataSet_index = sort(randsample(seed, tmin_ind:tmax_ind, points_per_run, points_per_run > len_t));  % sort only for comparison purposes
                    
                    DataSet(i,:,:) = cat(2, data_x(DataSet_index, :), after_jumps_label(DataSet_index));
                    %disp(sort(DataSet_index));
                    % disp(after_jumps_label);
                    % disp(after_jumps_label(DataSet_index));
                    
                end
                DataSet = reshape(permute(DataSet,[3, 1, 2]), this.state_dimension + 1, []);
                %DataSet = {after_jumps_label, sol}; %debug
            end
    end 
%data{1}(~isnan(data{1})) == data_test{1}(~isnan(data{1}))
% 
% figure(8)
% clf
% scatter(data_test{2}.t(data_test{1} == 0), data_test{2}.x(data_test{1} == 0,1), 8, 'red')
% hold on
% scatter(data_test{2}.t(data_test{1} == 1), data_test{2}.x(data_test{1} == 1,1), 8, 'blue')
% hold off
% scatter(data{2}.t(data{1} == 0), data{2}.x(data{1} == 0,1), 8, 'red')
% hold on
% scatter(data{2}.t(data{1} == 1), data{2}.x(data{1} == 1,1), 8, 'blue')
% hold off
% plot(data_test{1} - data{1})
end
   