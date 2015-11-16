function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

% First compute all prediction errors (costs) assuming every movie i was
% rated by every user j
all_predictions = X * Theta';
all_prediction_errors = all_predictions - Y;

% Then zero out all (i,j) except where R>0, and square
masked_all_prediction_errors = all_prediction_errors .* R;
masked_all_prediction_errors_squared = masked_all_prediction_errors .^ 2;
% Then get J but summing over the squares of the errors
% (don't forget to divide by 2, per the formula)
J = (1/2) * sum(sum(masked_all_prediction_errors_squared));

% Add regularization terms to J
J = J + (lambda/2) * sum(sum(X .^ 2)) + (lambda/2) * sum(sum(Theta .^ 2));

% The gradient of J with respect to X
for i=1:num_movies
	% Mask out Theta rows j (and Y columns j) where user j has not rated movie i
	mask_index = find(R(i,:)==1);
	masked_Theta = Theta(mask_index,:);
	masked_Y = Y(i,mask_index);
	X_grad(i,:) = ((X(i,:) * masked_Theta') - masked_Y) * masked_Theta;
end

% Add regularization term to X_grad
X_grad = X_grad + (lambda * X);

% The gradient of J with respect to Theta
for j=1:num_users
	% Mask out X rows i (and Y rows i) where user j has not rated movie i
	mask_index = find(R(:,j)==1);
	masked_X = X(mask_index,:);
	masked_Y = Y(mask_index,j);
	Theta_grad(j,:) = ((masked_X * Theta(j,:)') - masked_Y)' * masked_X;
end

% Add regularization term to Theta_grad
Theta_grad = Theta_grad + (lambda * Theta);

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
