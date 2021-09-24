function x_norm = normalize_features(x)
    %Use repmat to avoid loops
    %Calculate mean
    x_mean = repmat(mean(x), [length(x),1]);
    %Normalize
    x_norm = (x-x_mean) ./ std(x);
end