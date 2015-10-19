function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Create a column of 1s
column_of_1s = ones(m,1);

% Feedforward X through Theta1 and Theta2 (with sigmoids) to get h_sub_Theta_of_X
h_sub_Theta_of_X = sigmoid([column_of_1s , sigmoid([column_of_1s , X] * Theta1')] * Theta2');

% Convert y from a vector of labels to a matrix of (horizontal) vectors, where
% each y label becomes a vector with a 1 at the label's position and 0s elsewhere
I_num_labels = eye(num_labels);
vector_maker = @(label) I_num_labels(label,:);
vectorized_y = vector_maker(y);

% Compute the cost function (without regularization)
J = (1/m) * sum(sum(((((-vectorized_y).*log(h_sub_Theta_of_X)) - ((1.-vectorized_y).*log(1.-h_sub_Theta_of_X)))), 2));

% Add regularization - exclude bias (first) columns of Theta1 and Theta2
regularization_term = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));
J = J + regularization_term;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

% Initialize gradient accumulators to 0
Theta1_grad_accumulator = zeros(size(Theta1_grad));
Theta2_grad_accumulator = zeros(size(Theta2_grad));
% Accumulate gradients
for i = 1:m
	a_1 = X(i,:)';
	a_1 = [1 ; a_1];
	z_2 = Theta1 * a_1;
	a_2 = sigmoid(z_2);
	a_2 = [1 ; a_2];
	z_3 = Theta2 * a_2;
	a_3 = sigmoid(z_3);
	delta_3 = a_3 - vectorized_y(i,:)';
	delta_2 = Theta2' * delta_3;
	delta_2 = delta_2(2:end);
	delta_2 = delta_2 .* sigmoidGradient(z_2);
	Theta1_grad_accumulator = Theta1_grad_accumulator + (delta_2 * a_1'); % accumulate during loop, divide by m outside the loop
	Theta2_grad_accumulator = Theta2_grad_accumulator + (delta_3 * a_2');
end
% Divide accumulators by m to get gradients
Theta1_grad = Theta1_grad_accumulator / m;
Theta2_grad = Theta2_grad_accumulator / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Add regularization, but don't regularize the bias (first) columns
Theta1_regularization = (lambda/m) * Theta1;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + Theta1_regularization(:,2:end);

Theta2_regularization = (lambda/m) * Theta2;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + Theta2_regularization(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
