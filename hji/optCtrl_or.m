
function uOpt = optCtrl_or(obj, ~, ~, V, dV, uMode)
  % specific to a DynSys
  % double integrator 2D
  % compute the optimal control for phi & psi. If such a controller doesn't exist, then it will choose the optimal control that violates both formulas the least.
  % TODO: bias one of the formulas if both are infeasible.

    n = obj.nu;
    u = zeros(2,1);
    
    if strcmp(uMode, 'min')

        [~, ind] = min(V);
        deriv = dV{ind}([2;4])';

        u(1) = (deriv(1)>=0)*obj.uMin + (deriv(1)<0)*obj.uMax;
        u(2) = (deriv(2)>=0)*obj.uMin + (deriv(2)<0)*obj.uMax;    
        
        
    elseif strcmp(uMode, 'max')

        [~, ind] = max([V_phi, V_psi]);
        deriv = dV{ind}([2;4])';

        u(1) = (deriv(1)<0)*obj.uMin + (deriv(1)>=0)*obj.uMax;
        u(2) = (deriv(2)<0)*obj.uMin + (deriv(2)>=0)*obj.uMax;  
    else
        error('Unknown uMode!')
    end
    
    uOpt = u;
end
