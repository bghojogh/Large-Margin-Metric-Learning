function triplet_indices = Calculate_eta_and_triplet_indices(X_train,y_train,k, n_train, y_ij, lambda_, method, method_index)
switch method(method_index)
    case 'k_batch_all'
        eta_friend = generate_eta_friend(X_train, y_train, k, 1);
        triplet_indices = triplet_generation(eta_friend, nan, n_train, y_ij, 1);
    case 'k_bath_hard'
        eta_friend = generate_eta_friend(X_train, y_train, k, 0);
        eta_enemy = generate_eta_enemy(X_train, y_train, k, 1);
        triplet_indices = triplet_generation(eta_friend, eta_enemy, n_train, y_ij, 0);
    case 'k_batch_semi_hard'
        eta_friend = generate_eta_friend(X_train, y_train, k, 1);
        triplet_indices = triplet_generation_semi_hard(X_train, eta_friend, n_train, y_ij, k);
    case 'extreme_HPEN'
        eta_friend = generate_eta_friend(X_train, y_train, k, 0);
        eta_enemy = generate_eta_enemy(X_train, y_train, k, 0);
        triplet_indices = triplet_generation(eta_friend, eta_enemy, n_train, y_ij, 0);
    case 'extreme_EPEN'
        eta_friend = generate_eta_friend(X_train, y_train, k, 1);
        eta_enemy = generate_eta_enemy(X_train, y_train, k, 0);
        triplet_indices = triplet_generation(eta_friend, eta_enemy, n_train, y_ij, 0);
    case 'extreme_EPHN'
        eta_friend = generate_eta_friend(X_train, y_train, k, 1);
        eta_enemy = generate_eta_enemy(X_train, y_train, k, 1);
        triplet_indices = triplet_generation(eta_friend, eta_enemy, n_train, y_ij, 0);
    case 'negative_sampling'
        eta_friend = generate_eta_friend(X_train, y_train, k, 1);
        triplet_indices = triplet_generation_negative_sampling(X_train, eta_friend, n_train, y_ij, lambda_, k);
end
end

