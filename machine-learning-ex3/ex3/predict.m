function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Forward propagate X via Theta1 and Theta2 to get the output layer values (i,j) is the probability that X(i,:) is of label j
% Forward propagate via Theta1 (don't forget to add a column of 1's)
hidden_layer_values = sigmoid([ones(m, 1) , X] * Theta1')
% Forward propagate via Theta2 (don't forget to add a column of 1's)
output_layer_values = sigmoid([ones(m, 1) , hidden_layer_values] * Theta2')

% For each row in the output layer values, the max column is the likeliest label
[max_probabilities, likeliest_labels] = max(output_layer_values, [], 2)

p = likeliest_labels

% =========================================================================

end
