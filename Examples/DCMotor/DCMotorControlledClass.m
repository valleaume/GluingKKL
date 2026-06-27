classdef DCMotorControlledClass < HybridSubsystem
    % DCMotorHybridSystemClass: A hybrid system model of a DC motor with friction.
    % This class includes electrical dynamics (armature circuit) and mechanical dynamics
    % (rotor inertia, Coulomb/viscous friction, and load torque).

    properties
        % Electrical parameters
        R = 2.35;      % Armature resistance (Ohms)   9.3?
        L = 2.2e-3;      % Armature inductance (H)
        Kt = 0.28;     % Torque constant (Nm/A)
        Ke = 0.017;     % Back-EMF constant (V/(rad/s))

        % Mechanical parameters
        J = 8.3e-5;       % Rotor inertia (kg*m^2)
    
        T_load = 0.0;   % External load torque (Nm) (can be set dynamically)

        % Stiction threshold (to avoid division by zero)
        omega_threshold = 1e-3;
    end

    properties(SetAccess = immutable)
        % Indices for state variables: [current; angular velocity; angular position; mode; u; u_dot; F_s; F_d]
        current_index = 1;
        omega_index = 2;
        theta_index = 3;
        mode_index = 4;
        internal_state_u_index = 5; % renamed to avoid confusion with input u
        F_s_index = 6;
        F_d_index = 7;
    end

    methods
        function this = DCMotorControlledClass()
            % Constructor for DCMotorControlledClass.
            % Define state, input, and output dimensions.
            state_dim = 8;  % [current; omega; theta; mode; u; u_dot; F_s; F_d]
            input_dim = 1;  % Input voltage (unused when u is part of the state)
            output_dim = 1; % Output: angular position (theta)
            output_map = @(x) x(3); % Output is the angular position
            this = this@HybridSubsystem(state_dim, input_dim, output_dim, output_map);
        end

        % Flow map: Defines continuous dynamics
        function xdot = flowMap(this, x, u, t, j)
            % Continuous dynamics of the DC motor.
            % Inputs:
            %   x: State vector [current; omega; theta; mode; u; u_dot; F_s; F_d]
            %   u: Input voltage (V) - unused when the voltage is part of the state
            %   t: Time (unused here)
            %   j: Hybrid time (unused here)
            % Output:
            %   xdot: Time derivative of the state vector

            % Unpack the state
            i = x(this.current_index);
            omega = x(this.omega_index);
            mode = x(this.mode_index);
            internal_state = x(this.internal_state_u_index);
            F_s = x(this.F_s_index);
            F_d = x(this.F_d_index);

            % Electrical dynamics: di/dt = (u_state - R*i - Ke*omega) / L
            di_dt = (u_state - this.R * i - this.Ke * omega) / this.L;

            % Mechanical dynamics: domega/dt = (Kt*i - F_coulomb - T_load) / J
            % Coulomb friction is modeled as a sign-dependent torque
            if mode == 1 % Friction region
                T_friction = -F_d * sign(omega); % Dynamic friction opposes motion
                domega_dt = (this.Kt * i + T_friction) / this.J;
            else
                domega_dt = 0; % Stiction region: no motion (omega = 0) if current is below threshold
            end
        
            % Integrate angular velocity to get position: dtheta/dt = omega
            dtheta_dt = omega;

            if abs(u_state) > this.U_max
                u_dot = 0; % Saturate voltage dynamics to prevent unbounded growth
            end

            % Voltage dynamics: u_dot is part of the state and its derivative is zero
            du_dt = u_dot;
            dudot_dt = 0;

            % Friction parameters are treated as constant states
            dF_s_dt = 0;
            dF_d_dt = 0;

            dmode_dt = 0; % Mode does not flow

            % Return the state derivatives
            xdot = [di_dt; domega_dt; dtheta_dt; dmode_dt; du_dt; dudot_dt; dF_s_dt; dF_d_dt]; 
        end

        % Jump map: Defines discrete transitions 
        function xplus = jumpMap(this, x, u, t, j)
            % No discrete transitions for this system aside from mode change.
            mode = x(this.mode_index);
            mode_plus = 1-mode;
            xplus = x;
            xplus(this.mode_index) = mode_plus;
            if mode == 1 % Friction region
                xplus(this.omega_index) = 0; % Reset angular velocity to zero when transitioning to stiction
            end
        end

        % Flow set indicator: Uses the state-embedded static friction threshold
        function inC = flowSetIndicator(this, x, u, t, j)
           
            mode = x(this.mode_index);
            i = x(this.current_index);
            F_s = x(this.F_s_index);
            if mode == 0 % Stiction region
                inC = abs(this.Kt*i) < F_s; % Stays in stiction if current is below static threshold
            else
                inC = true;
            end
        end

        % Jump set indicator: Uses the state-embedded friction values
        function inD = jumpSetIndicator(this, x, u, t, j)
            mode = x(this.mode_index);
            i = x(this.current_index);
            omega = x(this.omega_index);
            F_s = x(this.F_s_index);
            F_d = x(this.F_d_index);

            if mode == 0 % Stiction region
                inD = abs(this.Kt*i) >= F_s; % Jumps to friction region if current exceeds static threshold
            else
                inD = (abs(this.Kt*i) < F_d) && (abs(omega) <= this.omega_threshold); % Jumps to stiction if current is below dynamic threshold and rotor is stationary
            end
        end
    end
end