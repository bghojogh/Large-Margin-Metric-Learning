function [data_subset, labels_subset,y_ij_subset,triplet_indices_subset,is_sample_within_hypersphere,is_sample_selected_in_hypersphere] = hypersphere_sample_points(center,radius,sampling_portion, data,labels,k,lambda_, method, method_index, do_sample_within_hypesphere)

n_samples = size(data, 1);
distances_from_center = my_norm(data(:,:)-repmat(center, n_samples,1), 2);
is_sample_within_hypersphere = zeros(n_samples, 1);
is_sample_within_hypersphere(distances_from_center <= radius, : ) = 1;

if do_sample_within_hypesphere
    %%%% stratified sampling:
    indices_selected_in_hypersphere = [];
    available_classes = unique(labels);
    for class_index = available_classes'
        is_sample_in_this_class = (labels == class_index);
        is_sample_in_this_class_and_within_hypersphere = is_sample_in_this_class & is_sample_within_hypersphere;
        indices_of_class_hypersphere_samples_in_dataset = find(is_sample_in_this_class_and_within_hypersphere == 1);
        n_samples_inClass_and_inHypersphere = length(indices_of_class_hypersphere_samples_in_dataset);
        if n_samples_inClass_and_inHypersphere == 0  %--> if not any point from this class in the hypersphere
            continue
        end
        n_samples_inClass_and_inHypersphere_sampled = max(ceil(sampling_portion * n_samples_inClass_and_inHypersphere), 1);
        indices_randomPermutation = indices_of_class_hypersphere_samples_in_dataset(randperm(length(indices_of_class_hypersphere_samples_in_dataset)));
        indices_sampled = indices_randomPermutation(1:n_samples_inClass_and_inHypersphere_sampled);
        indices_selected_in_hypersphere = [indices_selected_in_hypersphere; indices_sampled];
    end
    is_sample_selected_in_hypersphere = zeros(n_samples, 1);
    is_sample_selected_in_hypersphere(indices_selected_in_hypersphere) = 1;
else
    is_sample_selected_in_hypersphere = is_sample_within_hypersphere;
end

data_subset = data(is_sample_selected_in_hypersphere==1, :);
labels_subset = labels(is_sample_selected_in_hypersphere==1);
n_samples_subset = size(data_subset, 1);

if isempty(data_subset)
    y_ij_subset = [];
    triplet_indices_subset = [];
    is_sample_selected_in_hypersphere = zeros(n_samples, 1);
    disp("There is no point in this hypersphere");
elseif var(labels_subset)==0
    y_ij_subset = [];
    triplet_indices_subset = [];
    is_sample_selected_in_hypersphere = zeros(n_samples, 1);
    disp("Hypersphere is pure");
elseif n_samples_subset < 3
    y_ij_subset = [];
    triplet_indices_subset = [];
    is_sample_selected_in_hypersphere = zeros(n_samples, 1);
    disp("Hypersphere does not contain a triplet"); 
elseif n_samples_subset <= k
    y_ij_subset = [];
    triplet_indices_subset = [];
    is_sample_selected_in_hypersphere = zeros(n_samples, 1);
    disp("The number of points in the hypersphere is less than k"); 
else
    %eta_friend_subset = generate_eta_friend(data_subset, labels_subset, k, 1);
    y_ij_subset = generate_y_ij(labels_subset);
    y_ij_subset_elementwise_transpose = y_ij_subset.';
    upper_triangle_y_ij_subset_indices  = (1:size(y_ij_subset_elementwise_transpose,1)).' > (1:size(y_ij_subset_elementwise_transpose,2));
    upper_triangle_y_ij_subset_vector  = y_ij_subset_elementwise_transpose(upper_triangle_y_ij_subset_indices);
    
    if var(upper_triangle_y_ij_subset_vector) == 0
        y_ij_subset = [];
        triplet_indices_subset = [];
        disp("There is no either negative or positive in the hypersphere");
    else
        %triplet_indices_subset = triplet_generation(eta_friend_subset, nan, n_samples_subset, y_ij_subset, 1);
        triplet_indices_subset = Calculate_eta_and_triplet_indices(data_subset,labels_subset,k, n_samples_subset, y_ij_subset, lambda_, method, method_index);
    end
end
    
    
end

function norm_=my_norm(matrix_,along_dim)
    norm_= sqrt(sum(matrix_.^2, along_dim));
end

