function J = compute_cost_logistic_regression(x_norm, y_train, theta)
%Hypothesis
sig_H = compute_sigmoid(theta*x_norm');
%m
m = length(y_train);
%Calculate Cost
J= ((log(sig_H) * -y_train) - (log(1 - sig_H) * (1 - y_train)))*(1/m);

end