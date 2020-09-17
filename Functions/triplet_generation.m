function triplet_indices = triplet_generation(eta_friend, eta_enemy, n_train, y_ij, take_all_neg)

triplet_indices=[];
for i=1:n_train
    for j=1:n_train
        if eta_friend(i,j) ==1
            for k=1:n_train
                if take_all_neg == 1
                    if y_ij(i,k) == 0
                        triplet_indices = [triplet_indices; i, j, k];
                    end
                else
                    if eta_enemy(i,k) == 1
                        triplet_indices = [triplet_indices; i, j, k];
                    end
                end
            end
        end
    end
end

end

