function J = computeCostMulti(X, y, theta)
  
m = length(X);

J = 0;

h = X*theta;

error = (h-y).^2;

J = (1/(2*m))*sum(error);

end