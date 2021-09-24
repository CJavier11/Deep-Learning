function J = compute_cost_logistic_regression_regularized(x_norm, y_train, theta, reg_p)
%Hypothesis
sig_H = compute_sigmoid(theta*x_norm');
%m
m = length(y_train);
%Calculate Cost

J = (-1/m)* sum(log(sig_H)*y_train + (log(1 - sig_H) * (1 - y_train))) + reg_p/(2*m) * sum(theta(2:size(theta)).^2);

%J= (-y_train' .* log(sig_H) - (1 - y_train)' .* log(1 - sig_H)) ./ m + (reg_p/(2*m)) .* sum(theta(2:end).^2);

end