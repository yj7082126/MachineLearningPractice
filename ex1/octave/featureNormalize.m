function [X_norm, mu, sigma] = featureNormalize(X)
 
X_norm = X;
mu = mean(X);
sigma = std(X);
X_norm = (X_norm - mu)./sigma;



end
