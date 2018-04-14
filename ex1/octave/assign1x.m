%% Programming Exercise 1-2: Linear Regression (Multiple Variables)

%% Initialization
clear; 
close all; 
clc;

%% Data & Basic Variables
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

theta = zeros(3,1);

iterations = 50;
alpha = 1;

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% Feature Normalization
fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);
X = [ones(m, 1), X];

%% GradientDescent
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, iterations);

%% Plot (Convergence)
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

%% Price Estimation
predict = [1650, 3];
predict = (predict - mu)./sigma;
price = [1 predict] * theta;
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% Normal Equations
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

X = [ones(m, 1) X];

theta = pinv(X'*X)*X'*y;

price = [1 1650 3] * theta;
fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;


