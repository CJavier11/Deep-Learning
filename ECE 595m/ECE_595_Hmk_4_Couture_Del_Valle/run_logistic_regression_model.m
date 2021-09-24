%dataset
data = load('Sample Data.txt');

%Load x and y from sample data
x_train = data(:,1:end-1);
y_train = data(:,end);

%Plot x_train
%scatter(x_train(:,1), x_train(:,2))

%Normalize features and pad with ones 
x_norm = normalize_features(x_train);
x_norm = [ones(length(data),1),x_norm];

%variables
theta=zeros(1,size(x_train,2)+1);
reg_p = 10.24;
lr = 0.1;
n = 2000;

%Gradient descent function
[cost, theta_w, y_pred] = gradient_descent_logistic_regression_regularized(theta, x_norm, y_train, lr, n, reg_p);
%Print final cost
fprintf('\nFinal Cost:: %0.3f \n',cost(2000));
%Print theta 1,2,3
fprintf('\nTheta computed from gradient descent:: %0.4f \n',theta_w);

%Performance measure function
[acc_all0, c_mat, prec, rec, F1, spec] = performance_measure(y_pred, y_train);
%Print performance metrics
fprintf('\nOverall Accuracy:: %0.2f \n', acc_all0);
fprintf('\nTrue Positive:: %0.0f \n', c_mat(1,1));
fprintf('\nFalse Positive:: %0.0f \n', c_mat(1,2));
fprintf('\nFalse Negative:: %0.0f \n', c_mat(2,1));
fprintf('\nTrue Negative:: %0.0f \n', c_mat(2,2));
fprintf('\nPrecision:: %0.4f \n', prec(1));
fprintf('\nRecall:: %0.4f \n', rec(1));
fprintf('\nF1 Score:: %0.4f \n', F1(1));
fprintf('\nSpecificity:: %0.2f \n', spec(1));