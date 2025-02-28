classdef StickSlipSystemClassPerturbed < HybridSystem
    % A system exhibiting StickSlip behaviors modeled as a HybridSystem subclass.
    % It can be modelized as a mass m attached to a spring of stiffness k on a treadmill.
    
    % Define variable properties that can be modified.
    properties
        g = 9.8;  % Acceleration due to gravity.
        w = sqrt(10);  % Oscillator's pulsation, sqrt(k/m) for mechanical oscillator
        v_t = 0.5; % Speed of the treadmill
        time_of_perturbation = 80;
        perturbed = false;
        to_change = false;
         
    end
    
    % Define constant properties that cannot be modified (i.e., "immutable").
    properties(SetAccess = immutable) 
        % The index of 'position' component 
        % within the state vector 'x'. 
        position_index = 1;
        
        % The index of 'velocity' component 
        % within the state vector 'x'. 
        velocity_index = 2;

        % The index of '\mu_s' component, static friction coefficient,
        % within the state vector 'x'.
        % In order to be able to observe it, it must be part of the state
        % even though it could be seen as a "parameter".
        mu_s_index = 3;

        % The index of '\mu_d' component, dynamic friction coefficient,
        % within the state vector 'x'.
        % In order to be able to observe it, it must be part of the state
        % even though it could be seen as a "parameter".
        mu_d_index = 4;

        % The index of the 'q' state component, 
        % indicating whether the mass is sticking or moving compared to the treadmill.
        % q = sign( v_mass - v_treadmill), with sign(0) = 0
        q_index = 5;
    end
    methods 
        function this = StickSlipSystemClassPerturbed()
            % Constructor for instances of the StickSlipSystemClass.
            % Call the constructor for the HybridSystem superclass and
            % pass it the state dimension. This is not strictly necessary, 
            % but it enables more error checking.
            state_dim = 5;
            this = this@HybridSystem(state_dim);
        end
        % To define the data of the system, we implement 
        % the abstract functions from HybridSystem.m
        function xdot = flowMap(this, x, t, j)
            % Extract the state components.
            v = x(this.velocity_index);
            pos = x(this.position_index);
            q = x(this.q_index);
            mu_d = x(this.mu_d_index);

            % Define the value of the flow map f(x). 
            % mu_s, mu_d and q do not change during the flow period.
            xdot = [v; -abs(q)*this.w^2*pos - q*mu_d*this.g; 0; 0; 0];
        end
        function xplus = jumpMap(this, x, t, j)
            % Extract the state components.
            pos = x(this.position_index);
            v = x(this.velocity_index);
            q = x(this.q_index);
            mu_s = x(this.mu_s_index);
            mu_d = x(this.mu_d_index);

            % Define the value of jump map g(x). 
            if this.to_change
                mu_s = mu_s/2;
                mu_d = mu_d/2;
                this.to_change = false;
            end
            if q == 0
                q_new = -sign(pos);
            else
                if this.w^2*abs(pos) <= mu_d*this.g  
                    % If the force of the spring can be compensated by the dynamic friction force then the system stick
                    q_new = 0;
                    v = this.v_t;   % force v = v_t, help for latter jump detection
                else
                    % If the force of the spring is too much then the system cannot stick
                    q_new = -q;
                end
            end
            xplus = [pos; v; mu_s; mu_d; q_new];  
        end
        
        function inC = flowSetIndicator(this, x, t, j)
            % Extract the state components.
            pos = x(this.position_index);
            v = x(this.velocity_index);
            q = x(this.q_index);
            mu_s = x(this.mu_s_index);
        
            % Set 'inC' to 1 if 'x' is in the flow set and to 0 otherwise.
            if q == 0
                inC = (abs(pos) <= mu_s*this.g/this.w^2); % as long as the spring force can be compensated by the static friction force then the system stick
            else
                if q == 1
                    inC = (v>= this.v_t);
                else
                    if q == -1
                        inC = (v<= this.v_t);
                    else
                        inC = false;
                    end
                end
            end
        end


        function inD = jumpSetIndicator(this, x, t, j)
            % Extract the state components.
            pos = x(this.position_index);
            v = x(this.velocity_index);
            q = x(this.q_index);
            mu_s = x(this.mu_s_index);
            mu_d = x(this.mu_d_index);
            % Set 'inD' to 1 if 'x' is in the jump set and to 0 otherwise.
            if (t > this.time_of_perturbation) && (not (this.perturbed))
                this.perturbed = true;
                this.to_change = true;
                inD = true;
            end
            if q == 0
                inD = (abs(pos) >= mu_s*this.g/this.w^2) & (v == this.v_t);  
            else
                if q == 1
                    inD = (v <= this.v_t) & (-abs(q)*this.w^2*pos - q*mu_d*this.g) <= 0;
                else
                    if q == -1
                        inD = (v >= this.v_t) & (-abs(q)*this.w^2*pos - q*mu_d*this.g) >= 0;
                    else
                        inD = false;
                    end
                end
            end
        end
    end
end