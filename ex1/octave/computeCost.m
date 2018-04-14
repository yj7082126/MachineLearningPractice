function J = computeCost(data, theta)
  
m = length(data);
X = [ones(m,1), data(:,1)];
y = data(:,2);

J = 0;

h = X*theta;

error = (h-y).^2;

J = (1/(2*m))*sum(error);

end