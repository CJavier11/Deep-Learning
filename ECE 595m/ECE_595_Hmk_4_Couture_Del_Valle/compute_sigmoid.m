function z_sigmoid = compute_sigmoid(z)
    %z_sig
    z_sigmoid = 1./(1+exp(-z));
end