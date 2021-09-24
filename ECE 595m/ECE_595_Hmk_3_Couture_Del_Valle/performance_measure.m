function [acc_all0, c_mat, prec, rec, F1, spec] = performance_measure(y_pred, y_train)
%Set all values in y_pred above 0.5 to 1 and less than 0.5 to 0
y_pred( y_pred < 0.5 ) = 0;
y_pred( y_pred > 0.5 ) = 1;

%Initialize confusion matrix
c_mat = zeros(1,4);
%Calculate overall accuracy
acc_all0 = mean(double(y_pred' == y_train));

%Actual positives
ap = sum(y_train == 1);
%Actual negatives
an = sum(y_train == 0);

%True positives
tp = sum((y_pred' == 1) & (y_train == 1));
    c_mat(1) = tp;
%False positives
fp = sum((y_pred' == 1) & (y_train == 0));
    c_mat(2) = fp;
%False negatives
fn = sum((y_pred' == 0) & (y_train == 1));
    c_mat(3) = fn;
%True negatives
tn = sum((y_pred' == 0) & (y_train == 0));
    c_mat(4) = tn;
    
%Specificity
spec = tn/an;
%Precision
prec = tp / (tp + fp);
%Recall
rec = tp / (tp + fn);
%F1 Score
F1 = 2 * prec * rec / (prec + rec);


end