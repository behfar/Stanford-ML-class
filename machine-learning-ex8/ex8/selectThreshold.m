function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    % For each epsilon candidate, compute precision, recall, and F1

    % positives and negatives from the validation set
    yval_positives = (yval == 1);
    yval_negatives = (yval == 0);

    % positives and negatives as predicted by the current epsilon candidate
    epsilon_positives = (pval < epsilon);
    epsilon_negatives = (pval >= epsilon);

    % helper variables to get to precision, recall, and F1
    true_positives = (epsilon_positives .* yval_positives);
    false_positives = (epsilon_positives .* yval_negatives);
    true_negatives = (epsilon_negatives .* yval_negatives);
    false_negatives = (epsilon_negatives .* yval_positives);

    num_true_positives = sum(true_positives);
    num_false_positives = sum(false_positives);
    num_true_negatives = sum(true_negatives);
    num_false_negatives = sum(false_negatives);

    % precision, recall, and F1
    precision = num_true_positives / (num_true_positives + num_false_positives);
    recall = num_true_positives / (num_true_positives + num_false_negatives);
    F1 = 2 * (precision * recall) / (precision + recall);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
