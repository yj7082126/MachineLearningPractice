%% Programming Exercise 1: Linear Regression

%% Initialization
clear; 
close all; 
clc;

%% Data & Basic Variables
data = load('ex1data1.txt');
theta = zeros(2,1)

iterations = 1500;
alpha = 0.01;

%% Cost functions for examples
fprintf('\nTesting the cost function ...\n');
J = computeCost(data, theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 32.07\n');

J = computeCost(data, [-1 ; 2]);
fprintf('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 54.24\n');
pause;

%% Gradient Descent
fprintf('\nRunning Gradient Descent ...\n')
theta = gradientDescent(data, theta, alpha, iterations);

fprintf('Theta found by gradient descent:\n');
fprintf('%f\n', theta);
fprintf('Expected theta values (approx)\n');
fprintf(' -3.6303\n  1.1664\n\n');
pause;

%% Plot (Theta, Linear fit)
figure;
plot(data(:,1), data(:,2), 'rx', 'MarkerSize', 10);
ylabel('Profit in $10,000s');
xlabel('Population of City in 10,000s');
hold on;
plot(data(:,1), [ones(length(data),1), data(:,1)]*theta, '-')
legend('Training data', 'Linear regression');
hold off
pause;

%% Plot (Cost -- Surface, Contour)
fprintf('Visualizing J ... \n');

theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);

J_vals = zeros(length(theta0_vals), length(theta1_vals));

for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
      t = [theta0_vals(i); theta1_vals(j)];
      J_vals(i,j) = computeCost(data, t);
    end
end

J_vals = J_vals';

figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0');
ylabel('\theta_1');

figure;
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
