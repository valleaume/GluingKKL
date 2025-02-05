classdef StickSlipSystemClass < HybridSystem
    % A bouncing ball modeled as a HybridSystem subclass.
    % Define variable properties that can be modified.
    properties
        g = 9.8;  % Acceleration due to gravity.
        w = sqrt(10);  %Oscillator's pulsation, sqrt(k/m) for mechanical oscillator
        v_t = 0.5;
         
    end
    % Define constant properties that cannot be modified (i.e., "immutable").
    properties(SetAccess = immutable) 
        % The index of 'height' component 
        % within the state vector 'x'. 
        position_index = 1;
        
        % The index of 'velocity' component 
        % within the state vector 'x'. 
        velocity_index = 2;
        mu_s_index = 3;
        mu_d_index = 4;
        q_index = 5;
    end
    methods 
        function this = StickSlipSystemClass()
            % Constructor for instances of the BouncingBall class.
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

            % Define the value of f(x). 
            xdot = [v; -abs(q)*this.w^2*pos - q*mu_d*this.g; 0; 0; 0];
        end
        function xplus = jumpMap(this, x, t, j)
            % Extract the state components.
            pos = x(this.position_index);
            v = x(this.velocity_index);
            q = x(this.q_index);
            mu_s = x(this.mu_s_index);
            mu_d = x(this.mu_d_index);
            % Define the value of g(x). 

            if q == 0
                q_new = -sign(pos);
            else
                if this.w^2*abs(pos) <= mu_d*this.g
                    q_new = 0;
                    v = this.v_t;
                else
                    q_new = -q;
                end
            end
            xplus = [pos; v; mu_s; mu_d; q_new];  %v_t au lieu de v, facilite jump conditions
        end
        
        function inC = flowSetIndicator(this, x, t, j)
            % Extract the state components.
            pos = x(this.position_index);
            v = x(this.velocity_index);
            q = x(this.q_index);
            mu_s = x(this.mu_s_index);
            %mu_d = x(this.mu_d_index);
            % Set 'inC' to 1 if 'x' is in the flow set and to 0 otherwise.
            if q == 0
                inC = (abs(pos) <= mu_s*this.g/this.w^2); %& (v == this.v_t), autorise de la marge, même si non physique
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
            % Set 'inC' to 1 if 'x' is in the flow set and to 0 otherwise.
            if q == 0
                inD = (abs(pos) >= mu_s*this.g/this.w^2) & (v == this.v_t); %retire v == v_t?  
            else
                if q == 1
                    inD = (v<= this.v_t) & (-abs(q)*this.w^2*pos - q*mu_d*this.g) <= 0;
                else
                    if q == -1
                        inD = (v>= this.v_t) & (-abs(q)*this.w^2*pos - q*mu_d*this.g) >= 0;
                    else
                        inD = false;
                    end
                end
            end
        end
    end
end