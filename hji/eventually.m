function [data, tau2] = eventually(g, obj, tau, data0, target, visualize_args)
    % data, or target (time varying): negative values mean it is satisfied
    uMode = 'min';
    dMode = 'max';
    
    if nargin >= 5
        HJIextraArgs.targetFunction = target;
        HJIextraArgs.visualize.targetSet = false; 

        compMethod = 'maxVWithTarget';
    else
        compMethod = 'minVWithV0';
    end
    
    schemeData.grid = g;
    schemeData.dynSys = obj;
    schemeData.accuracy = 'high'; %set accuracy
    schemeData.uMode = uMode;
    schemeData.dMode = dMode;
    
    if nargin < 6
        HJIextraArgs.visualize.valueSet = 1;
        HJIextraArgs.visualize.initialValueSet = 1;
        HJIextraArgs.visualize.figNum = 1; %set figure number
        HJIextraArgs.visualize.deleteLastPlot = true; %delete previous plot as you update
    else
        HJIextraArgs.visualize = visualize_args;
    end
    [data, tau2, ~] = HJIPDE_solve(data0, tau, schemeData, compMethod, HJIextraArgs);

end