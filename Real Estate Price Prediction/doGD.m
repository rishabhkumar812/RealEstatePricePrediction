function [theta,recordJ] = doGD(theta,X,y)
  
  m = size(X,1);
  alpha = 0.0007;
  iter= 70000;
  recordJ = calcCostFunction(X,theta,y);
 
  for i = 1:iter
    h = calcH(theta,X);
    d = h-y;
    theta = theta - alpha * (X'*d)/m;
    J = calcCostFunction(X,theta,y);
    recordJ = [recordJ; J];
  endfor
  
  
end