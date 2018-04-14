function [theta, J_history] = gradientDescent(data, theta, alpha, num_iters)
  
m = length(data);
X = [ones(m,1), data(:,1)];
y = data(:,2);

J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    h = X*theta;
    theta0 = theta(1) - alpha*(1/m)*sum(h-y);
    theta1 = theta(2) - alpha*(1/m)*sum((h-y).*X(:,2));
    theta = [theta0; theta1];
    
    J_history(iter) = computeCost(data, theta);
end

end