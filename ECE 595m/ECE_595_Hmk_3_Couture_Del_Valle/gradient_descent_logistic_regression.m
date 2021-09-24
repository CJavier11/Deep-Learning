function [J, theta, h] = gradient_descent_logistic_regression(theta, x_norm, y_train, lr, n)
     %training samples
     m = length(y_train); 
     %Initialize J
     J = zeros(1,n);
     %Epochs
     for idx=1:n
        % Hypothesis
        h = compute_sigmoid(theta * x_norm');
        
        % Compute the cost for every iteration
        J(idx)=compute_cost_logistic_regression(x_norm,y_train, theta);
        %temp = compute_cost_logistic_regression(x_norm,y_train, theta);
        
        % Update weight
        dtheta=(1/m)*(h'-y_train)'*x_norm;
    
        % Update Theta
        theta=theta-lr*dtheta;
        %keyboard
     end
     %Plot
     plot(1:n, J)
end  