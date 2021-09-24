function [J, theta, h] = gradient_descent_logistic_regression_regularized(theta, x_norm, y_train, lr, n, reg_p)
     %training samples
     m = length(y_train); 
     %Initialize J
     J = zeros(1,n);
     %Epochs
     for idx=1:n
        % Hypothesis
        h = compute_sigmoid(theta * x_norm');
        
        % Compute the cost for every iteration
        J(idx)=compute_cost_logistic_regression_regularized(x_norm,y_train, theta, reg_p);
        %theta(0)
        theta(1)=theta(1)-(lr/m)*(h'-y_train)'*x_norm(:,1);
        %theta(j)
        theta(2:end)= theta(2:end)*((1-(lr*reg_p)/m))-(lr/m)*(h'-y_train)'*x_norm(:,2:end);
     end
     %Plot
     plot(1:n, J)
end  