function [data, tau2] = until(g, obj, tau, data0, obstacle, target, visualize_args)
    % data, or target (time varying): negative values mean it is satisfied
    % obstacle: negative values mean not satisfied, i.e., want to avoid
    % that set.
    uMode = 'min';
    dMode = 'max';
    
    HJIextraArgs.obstacleFunction = obstacle;
    
    if nargin >= 6
        HJIextraArgs.targetFunction = target;
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
        HJIextraArgs.visualize.targetSet = true; 
    else
        HJIextraArgs.visualize = visualize_args;
    end
    [data, tau2, ~] = HJIPDE_solve(data0, tau, schemeData, 'minVWithTarget', HJIextraArgs);

end