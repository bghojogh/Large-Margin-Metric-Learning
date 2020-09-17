function triplet_indices = triplet_generation_semi_hard(X_train, eta_friend, n_train, y_ij, K)

[mIdx,knn_distance] = knnsearch(X_train,X_train,'K',n_train);
triplet_indices=[];
for i=1:n_train
    for j=1:n_train
        if eta_friend(i,j) ==1
            counter_enemy = 0;
            for k=1:n_train
                if y_ij(i,mIdx(k)) == 0 && knn_distance(i,k) >= norm(X_train(i,:)-X_train(j,:),2)
                    counter_enemy = counter_enemy +1;
                    triplet_indices = [triplet_indices; i, j, mIdx(k)];
                end
                if counter_enemy >= K
                    break
                end
            end
        end
    end
end

end

