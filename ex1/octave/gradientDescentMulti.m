function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
  
m = size(X, 1);
n = size(X, 2);


J_history = zeros(num_iters, 1);

for iter = 1:num_iters
  theta_tmp = theta;
  h = X*theta_tmp;
  for element = 1:n   
    theta_tmp(element) = theta(element) - alpha*(1/m)*sum(X(:,element)'*(h-y));
  end
  theta = theta_tmp;
  J_history(iter) = computeCostMulti(X, y, theta);
    
end

end