function [cost, theta] = gradient_descent_lr_multi_variables(theta, x_norm, y_train, lr, n)
     %training samples
     m = length(y_train); 
     
     %Epochs
     for iter = 1:n
         
        %Calculate hypothesis
        h = x_norm * theta';
        
        %Gradient descent calculation
        for i=1:size(x_norm,2)
            theta(i) = theta(i) - lr/m * sum((h-y_train) .* x_norm(:,i));
        end
        
        %Call cost function
        cost(iter) = compute_cost_mean_square_multi_variables(x_norm, y_train, theta);
        %Plot
        fprintf('\nCost: %f ', cost)
        plot(1:iter,cost)
        xlabel('Num Iterations');
        ylabel('Cost');
     end
end  