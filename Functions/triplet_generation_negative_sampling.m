function triplet_indices = triplet_generation_negative_sampling(X_train, eta_friend, n_train, y_ij, lambda_, K)

n_features = size(X_train,2);
[mIdx,knn_distance] = knnsearch(X_train,X_train,'K',n_train);
probability_negatives = zeros(n_train, n_train);
for i = 1:n_train
    for j=1:n_train
        if y_ij(i,j)==0
            distance = norm(X_train(i,:)-X_train(j,:),2);
            q_distance = (distance^(n_features-2))*((1 - 0.25*(distance^2))^((n_features-3)/2));
            probability_negatives(i,j) = min(lambda_, 1/q_distance);
        end
    end
    probability_negatives(i,:) = probability_negatives(i,:)/sum(probability_negatives(i,:));
end

triplet_indices=[];
for i=1:n_train
    for j=1:n_train
        if eta_friend(i,j) ==1
            retry_counter=0;
            enemy_indices_list = [];
            for distance_index = 1:K
                r = rand;
                an_enemy_is_sampled = false;
                for l=1:n_train
                    if r <= sum(probability_negatives(i,1:l)) || retry_counter >= 100
                        if ismember(l,enemy_indices_list)==0
                            enemy_indices_list = [enemy_indices_list,l];
                            an_enemy_is_sampled = true;
                            triplet_indices = [triplet_indices; i, j, l];
                            break
                        end
                    end
                    if ~an_enemy_is_sampled
                        retry_counter = retry_counter+1;
                        distance_index = distance_index-1;
                    end
                end
            end
        end
    end
end

end

