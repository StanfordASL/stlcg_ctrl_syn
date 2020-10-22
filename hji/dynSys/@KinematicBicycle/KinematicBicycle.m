classdef KinematicBicycle < DynSys
    % Double integrator class; subclass of DynSys (dynamical system)
    properties
        uMin    % Control bounds [a, delta]
        uMax
        dMin
        dMax
        dims    % Active dimensions
    end % end properties

    methods
        function obj = KinematicBicycle(x, umin, umax, dmin, dmax, dims)
            % DoubleInt(x, urange)
            %     Constructor for the double integrator
            %
            % Inputs:
            %     x - initial state (ignored in reachability computations; only used
            %         for simulation)
            %     urange - control bounds
            %     dims - active dimensions

            if nargin < 1
                x = [0; 0; 0; 0];
            end

            if nargin < 2
                umin = [-3, 0.344];
            end

            if nargin < 3
                umax = [3, 0.344];
            end

            if nargin < 4
                dmin = [-0.5, -0.5];
            end

            if nargin < 5
                dmax = [0.5, 0.5];
            end

            if nargin < 6
                dims = 1:4;
            end


            %% Basic properties for bookkeepping
            obj.pdim = [find(dims == 1) find(dims == 2)];
            %       obj.vdim = find(dims == 2);
            obj.nx = length(dims);
            obj.nu = 2;
            obj.nd = 2;

            %% Process input
            % Make sure initial state is 2D
            if numel(x) ~= 4
                error('Kinematic Bicycle state must be 4D.')
            end

            % Make sure initial state is a column vector
            if ~iscolumn(x)
                x = x';
            end

            obj.x = x;
            obj.xhist = x; % State history (only used for simulation)

            %% Process control range
            if numel(umin) ~= 2
                error('Control range must be 2D!')
            end
            if numel(umax) ~= 2
                error('Control range must be 2D!')
            end
            if numel(dmin) ~= 2
                error('Control range must be 2D!')
            end
            if numel(dmax) ~= 2
                error('Control range must be 2D!')
            end

            if (umax <= umin) > 0
                error('Control range vector must be strictly ascending!')
            end

            obj.uMin = umin;
            obj.uMax = umax;
            obj.dMin = dmin;
            obj.dMax = dmax;
            obj.dims = dims;
        end % end constructor
    end % end methods
end % end class