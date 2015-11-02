function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))

% try all pairs of following values for C and sigma
values_to_try = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
% accummulate triples of the form [C_candidate, sigma_candidate, model_error]
candidate_params_accum = [];
for C_candidate = values_to_try
	for sigma_candidate = values_to_try
		% train a model based on model params C and sigma
		model= svmTrain(X, y, C_candidate, @(x1, x2) gaussianKernel(x1, x2, sigma_candidate));
		% run model on cross-validation data
		predictions = svmPredict(model, Xval);
		% note the model's error on the cross-validation data
		model_error = mean(double(predictions ~= yval));
		% append the model params C and sigma and the model error to the end of the accumulators
		candidate_params_accum = [candidate_params_accum ; [C_candidate, sigma_candidate, model_error]];
	end
end
% the best model params are the ones with the smallest error (error is accumulated in column 3)
[best_candidate_error, best_candidate_index] = min(candidate_params_accum(:,3));
best_candidate = candidate_params_accum(best_candidate_index, :);
C = best_candidate(1); sigma = best_candidate(2);

% =========================================================================

end
