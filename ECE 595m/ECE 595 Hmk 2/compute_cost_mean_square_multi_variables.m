function cost = compute_cost_mean_square_multi_variables(x_norm, y_train, theta)

%training samples
m = length(y_train); 

%hypothesis h(x)
h = x_norm * theta'; 
%mse
sError = (h - y_train) .^ 2;  
%summation
cost = sum(sError) / (2 .* m); 

end