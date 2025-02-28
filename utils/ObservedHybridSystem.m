classdef ObservedHybridSystem < HybridSystem
    %OBSERVED HYBRID SYSTEM Wrapper for hybrid system object
    %   This class allows to add an observation function h to an object of
    %   the class hybrid system. System MUST be autonomous. h can be
    %   non-deterministic, ie subject to noise.
    properties
        def;
        h;
    end
    properties(SetAccess = immutable)
        ny;
        %state_dimension; immutable already defined
        unobservedSystem;
    end
    methods
        function this = ObservedHybridSystem(UnobservedSystem, ny, h)
            nx = UnobservedSystem.state_dimension;
            this = this@HybridSystem(nx);
            this.unobservedSystem = UnobservedSystem;
            %this.state_dimension = nx;
            this.ny = ny;
            if nargin == 3
                this.h = h;
                %h.setAccess = 'immutable';
                assert(this.check_h(h), "Wrong type for h, should be $\mathbb{R}^{this.nx} \times \mathbb{R} \mapsto \mathbb{R}^{this.ny}$")
            end
        end

        function set_h(this, h)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here, does not work if h
            %   immutable
            this.h = h;
            assert(this.check_h(h), "Wrong type for h, should be $\mathbb{R}^nx \times \mathbb{R} \mapsto \mathbb{R}^ny$")
        end
        
        %HERE system must be autonomous, If not there is a problem with
        %lambda?
        function Xdot = flowMap(this, x, t, j)
            Xdot = this.unobservedSystem.flowMap(x, t, j); %t, j
        end

        function Xplus = jumpMap(this, x, t, j)
            Xplus = this.unobservedSystem.jumpMap(x, t, j); % t, j
        end

        function inC = flowSetIndicator(this, x, t, j)
            inC = this.unobservedSystem.flowSetIndicator(x, t, j); % t, j
        end

        function inD = jumpSetIndicator(this, x, t, j)
            inD = this.unobservedSystem.jumpSetIndicator(x,t,j); % t, j  ?
        end

        function y = get_y(this, x)
            y = this.h(x);
        end

        function bool = check_h(this, h)
            %also check that it should be a function
            nx = this.state_dimension;
            bool = 1;%(length(h(ones(nx,1))) == this.ny);  %maybe ones(nx) not in C \cup D, should add a check here
        end
    end    
end    