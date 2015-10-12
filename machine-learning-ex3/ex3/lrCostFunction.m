function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

% Step 1: compute the unregularized [J, grad]

% first compute h sub theta of X
h_sub_theta_of_X = sigmoid(X * theta);

% plug it into the formula for J
J = (1/m) * sum((((-y).*log(h_sub_theta_of_X)) - ((1.-y).*log(1.-h_sub_theta_of_X))));

% and similarly plug it into the formula for grad
grad = (1/m) * (X' * (h_sub_theta_of_X - y));

% Step 2: now regularize [J, grad]

% first create a theta "mask", which is just like theta but with its first row zeroed out
theta_mask = [0 ; theta(2:end)];
% add regularization to J using the theta "mask" (i.e. don't regularize first row) 
J = J + ((lambda/(2*m)) * (theta_mask' * theta_mask));

% and add regularization to grad
grad = grad + ((lambda/m) * theta_mask);

% =============================================================

end
