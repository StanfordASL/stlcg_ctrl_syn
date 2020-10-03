function [data, tau2] = eventually(g, obj, tau, data0, target, compMethod, visualize_args)
    % data, or target (time varying): negative values mean it is satisfied
    uMode = 'min';
    dMode = 'max';
    
    if nargin >= 5
        HJIextraArgs.targetFunction = target;
    else
        HJIextraArgs.targetFunction = data0;
        target = data0;
    end
    HJIextraArgs.visualize.targetSet = false;
    
    if nargin < 6
        if length(size(target)) > length(size(data0))
            compMethod = 'maxVWithTarget';
        else
            compMethod = 'minVWithV0';
        end
    end
    schemeData.grid = g;
    schemeData.dynSys = obj;
    schemeData.accuracy = 'high'; %set accuracy
    schemeData.uMode = uMode;
    schemeData.dMode = dMode;
    
    if nargin < 7
        HJIextraArgs.visualize.valueSet = 1;
        HJIextraArgs.visualize.initialValueSet = 1;
        HJIextraArgs.visualize.figNum = 1; %set figure number
        HJIextraArgs.visualize.deleteLastPlot = true; %delete previous plot as you update
    else
        HJIextraArgs.visualize = visualize_args;
    end
    [data, tau2, ~] = HJIPDE_solve(data0, tau, schemeData, compMethod, HJIextraArgs);

end