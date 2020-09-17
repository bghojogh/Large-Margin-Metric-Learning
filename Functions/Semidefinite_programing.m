%%% some partss of this code are inspired by the code written by Alison Cheeseman and Jonah Ho

function [M, Xi] = Semidefinite_programing(X_train, triplet_indices, c)

%X_train is row-wise

x = X_train.';
N_features = size(X_train, 2);

cvx_begin %quiet
variable M(N_features, N_features) symmetric
variable Xi(size(triplet_indices,1)) nonnegative
sum_left = 0;
sum_right = 0;
for triplet_index = 1:size(triplet_indices,1)
    i = triplet_indices(triplet_index,1);
    j = triplet_indices(triplet_index,2);
    sum_left = sum_left + ((x(:,i)-x(:,j)).'*M*(x(:,i)-x(:,j)));
    sum_right = sum_right + Xi(triplet_index);
end
f0 = sum_left + c*sum_right;
minimize f0
subject to
Xi>=0;
M == semidefinite(N_features);
for triplet_index = 1:size(triplet_indices,1)
    i = triplet_indices(triplet_index,1);
    j = triplet_indices(triplet_index,2);
    l = triplet_indices(triplet_index,3);
    (x(:,i)-x(:,l)).'*M*(x(:,i)-x(:,l))-(x(:,i)-x(:,j)).'*M*(x(:,i)-x(:,j)) >= 1-Xi(triplet_index);
end
cvx_end

end

