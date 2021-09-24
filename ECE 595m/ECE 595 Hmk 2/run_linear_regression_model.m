%dataset
%data = load('airfoil_self_noise.txt');
data = load('Body Fat.txt');

%Load x and y bfat
x_train = data(:,[1,2]);
%disp(x_train)
y_train = data(:,3);
%disp(y_train);

%Load x and y airfoil
%x_train = data(:,(1:5));
%y_train = data(:,6);

%Normalize features and pad with ones Bfat
x_norm = ones(25,3);
x_norm(:,[2,3]) = normalize_features(x_train);
%disp(x_norm)
%Normalize features and pad with ones Airfoil
%x_norm = ones(1503,1);
%x_norm(:,(2:6)) = normalize_features(x_train);
%size(x_norm)
%disp(x_norm)

%variables
%theta = [0,0,0,0,0,0];
theta = [0,0,0];
lr = 0.05;
n = 300;

[cost, theta_w] = gradient_descent_lr_multi_variables(theta, x_norm, y_train, lr, n);
fprintf('\nTheta computed from gradient descent:\ntheta0: %f, \ntheta1: %f, \ntheta2: %f', theta_w(1), theta_w(2), theta_w(3));
%fprintf('\ntheta3: %f, \ntheta4: %f, \ntheta5: %f', theta_w(4), theta_w(5), theta_w(6));