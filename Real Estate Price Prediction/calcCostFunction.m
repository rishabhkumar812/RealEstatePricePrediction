function J = calcCostFunction(X,theta,y)
  
  h=calcH(theta,X);
  m = size(y,1);
  d = h-y;
  J = (d'*d)/(2*m);

end