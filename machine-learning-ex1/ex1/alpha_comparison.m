function [ ] = alpha_comparison(X,y,theta,alpha_start,num_alphas,num_iters)
%ALPHA_COMPARISON Summary of this function goes here
%   Detailed explanation goes here
fig = figure; hold on;
alpha = alpha_start;
for i=1:num_alphas
    [theta, J_history] = gradientDescentMulti(X,y,theta,alpha,num_iters);
    plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
    xlabel(alpha); 
    fprintf('alpha %d plotted. J = %d. Press enter for next alpha...\n', alpha, J_history(num_iters));
    pause;
    alpha = alpha * 1.5;
end
close(fig);
end

