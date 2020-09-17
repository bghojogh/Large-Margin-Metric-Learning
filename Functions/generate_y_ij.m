function y_ij = generate_y_ij(y_train)
N_train = length(y_train);
y_ij = zeros(N_train, N_train);
for i=1:N_train
    for j=1:N_train
        if y_train(i) == y_train(j)
            y_ij(i,j) = 1;
        end
    end
end
end

