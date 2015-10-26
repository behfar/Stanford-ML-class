function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Step 1: compute the unregularized [J, grad]

% first compute h sub theta of X
h_sub_theta_of_X = X * theta;

% plug it into the formula for J
errors = h_sub_theta_of_X - y;
J = (1/(2*m)) * (errors' * errors);

% and similarly plug it into the formula for grad
grad = (1/m) * (X' * errors);

% Step 2: now regularize [J, grad]

% first create a theta "mask", which is just like theta but with its first row zeroed out
theta_mask = [0 ; theta(2:end)];
% add regularization to J using the theta "mask" (i.e. don't regularize first row) 
J = J + ((lambda/(2*m)) * (theta_mask' * theta_mask));

% and add regularization to grad
grad = grad + ((lambda/m) * theta_mask);

% =========================================================================

grad = grad(:);

end
