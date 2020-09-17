%% Initialization

%clear command file
clc
%clear history
clear
% closes all the figures
close all

addpath('./Functions')


%% Settings

dataset ='iris';  %--> iris, wine, ORL_faces, MNIST
do_validation = 0;
split_dataset_again = 0;
method_index = 2;
lambda_ = 1.4;
do_sample_within_hypesphere = 1;
hierarchical_mode = 0;
log_results = 1;
make_dataset_2D = 0;
projectAllData_in_hierarchicalMode = 1;
if strcmp(dataset, 'iris') || strcmp(dataset, 'wine')
    n_iteration = 10;
elseif strcmp(dataset, 'ORL_faces')
%     n_iteration = 15;
    n_iteration = 30;
elseif strcmp(dataset, 'MNIST')
%     n_iteration = 30;
    n_iteration = 50;
end

method = ["k_batch_all", "k_bath_hard", "k_batch_semi_hard", "extreme_HPEN", "extreme_EPEN", "extreme_EPHN", "negative_sampling"];

if hierarchical_mode == 1
    path_save = sprintf('./saved_files/hierarchical/%s/', dataset);
else
    path_save = sprintf('./saved_files/original/%s/', dataset);
end
if ~exist(path_save, 'dir')
    mkdir(path_save);
end

if do_validation == 1
    if strcmp (dataset, 'wine') || strcmp (dataset, 'iris')
        %k_list = [3, 5, 7];
        k_list = [3, 5];
        c_list = [0.1, 0.5, 1];
    elseif strcmp (dataset, 'ORL_faces') || strcmp (dataset, 'MNIST')
        k_list = [1, 3, 5];
        c_list = [0.1, 0.5, 1];
    end
    
else
    k = 3;
    c = 0.1; % weight of hinge loss
end

%% Reading Datasets

switch dataset
    case 'iris'
        [data, txt] =  xlsread('./input/iris.csv') ;
        
        % making a column of zeros
        labels = zeros(length(txt),1);
        for i=1:length(txt)
            if strcmp(txt{i}, 'Iris-setosa')
                labels(i)=1;
            elseif strcmp(txt{i}, 'Iris-versicolor')
                labels(i)=2;
            elseif strcmp(txt{i}, 'Iris-virginica')
                labels(i)=3;
            end
        end
        
        if make_dataset_2D == 1
            PCA_projection_directions = pca(data);
            PCA_projection_directions = PCA_projection_directions(:,1:2);
            data = (PCA_projection_directions'*data')';
        end
        
        if hierarchical_mode == 1 && projectAllData_in_hierarchicalMode == 0
            data = data - mean(data, 1);
        end
        
    case 'wine'
        [data, txt] = xlsread('./input/wine.csv');
        labels = data(:,1);
        data = data(:,2:end);
        
        if make_dataset_2D == 1
            PCA_projection_directions = pca(data);
            PCA_projection_directions = PCA_projection_directions(:,1:2);
            data = (PCA_projection_directions'*data')';
            data = data - mean(data, 1);
        end
        
        if hierarchical_mode == 1 && projectAllData_in_hierarchicalMode == 0
            data = data - mean(data, 1);
        end
        
    case 'ORL_faces'
        %data = zeros(400,112*92);
        data = zeros(400,38*31);
        labels = zeros(400,1);
        for i=1:400
            downsample_images = imresize(imread(sprintf('./input/ORL/%d.jpg', i)),[38,31]);
            %downsample_images =imread(sprintf('./input/ORL/%d.jpg', i));
            data(i,:) = reshape(downsample_images, 1, []);
            labels(i) = floor((i-1)/10)+1;
        end
        data_no_pca = data;
        
        PCA_projection_directions = pca(data);
        if make_dataset_2D == 1
            PCA_projection_directions = PCA_projection_directions(:,1:2);
        else
            PCA_projection_directions = PCA_projection_directions(:,1:15);
        end
        data = (PCA_projection_directions'*data')';
        
        if hierarchical_mode == 1 && projectAllData_in_hierarchicalMode == 0
            data = data - mean(data, 1);
        end
        
    case 'MNIST'
        imgFile = '.\input\MNIST\train-images.idx3-ubyte';
        labelFile = '.\input\MNIST\train-labels.idx1-ubyte';
        readDigits_train = 400;
        [imgs_train labels_train] = readMNIST(imgFile, labelFile, readDigits_train, 0);
        imgFile = '.\input\MNIST\t10k-images.idx3-ubyte';
        labelFile = '.\input\MNIST\t10k-labels.idx1-ubyte';
        readDigits_test = 100;
        [imgs_test labels_test] = readMNIST(imgFile, labelFile, readDigits_test, 0);
        imgFile = './input/MNIST/train-images.idx3-ubyte';
        labelFile = './input/MNIST/train-labels.idx1-ubyte';
        readDigits_val = 100;
        [imgs_val labels_val] = readMNIST(imgFile, labelFile, readDigits_val, 500);
        
        X_train = zeros(readDigits_train,28*28);
        y_train = zeros(readDigits_train,1);
        for i=1:readDigits_train
            X_train(i,:) = reshape(imgs_train(:,:,i), 1, []);
            y_train(i) = labels_train(i);
        end
        X_test = zeros(readDigits_test,28*28);
        y_test = zeros(readDigits_test,1);
        for i=1:readDigits_test
            X_test(i,:) = reshape(imgs_test(:,:,i), 1, []);
            y_test(i) = labels_test(i);
        end
        X_val = zeros(readDigits_val,28*28);
        y_val = zeros(readDigits_val,1);
        for i=1:readDigits_val
            X_val(i,:) = reshape(imgs_val(:,:,i), 1, []);
            y_val(i) = labels_val(i);
        end
        
        data_no_pca = X_train;
        PCA_projection_directions = pca(X_train);
        if make_dataset_2D == 1
            PCA_projection_directions = PCA_projection_directions(:,1:2);
        else
            PCA_projection_directions = PCA_projection_directions(:,1:30);
        end
        X_train = (PCA_projection_directions'*X_train')';
        X_test = (PCA_projection_directions'*X_test')';
        X_val = (PCA_projection_directions'*X_val')';
        n_train = readDigits_train;
        n_test = readDigits_test;
        n_val = readDigits_val;
        
        if hierarchical_mode == 1 && projectAllData_in_hierarchicalMode == 0
            X_train = X_train - mean(X_train, 1);
            X_test = X_test - mean(X_train, 1);
            X_val = X_val - mean(X_train, 1);
        end
end

if strcmp (dataset, 'wine') || strcmp (dataset, 'iris')
    n_total = length(labels);
    n_train =  round(0.7*length(labels));
    n_test = n_total - n_train;
    
    train_indices = randperm(length(labels), n_train);
    X_train = zeros(n_train , size(data,2));
    X_test = zeros(n_test , size(data,2));
    y_train = zeros(n_train, 1);
    y_test = zeros(n_test, 1);
    j_train=0;
    j_test=0;
    
    for j=1:size(data,1)
        if ismember(j,train_indices)
            j_train=j_train+1;
            X_train(j_train,:) = data(j, :);
            y_train(j_train) = labels(j);
        else
            j_test=j_test+1;
            X_test(j_test, :)= data(j,:);
            y_test(j_test) = labels(j);
        end
    end
elseif strcmp (dataset, 'ORL_faces')
    if max(max(data)) > 1
        data = data./255;
    end
    X_train=[];
    X_val=[];
    X_test=[];
    y_train=[];
    y_val=[];
    y_test=[];
    X_train_no_pca=[];
    X_test_no_pca=[];
    
    labels = reshape(labels, [], 1);
    for class_index = 1:length(unique(labels))
        X_this_class = data(labels == class_index, :);
        X_this_class_no_pca = data_no_pca(labels == class_index, :);
        labels_this_class = labels(labels == class_index);
        X_train = [X_train; X_this_class(1:6, :)];
        X_val = [X_val; X_this_class(7:8, :)];
        X_test = [X_test; X_this_class(9:10, :)];
        y_train = [y_train; labels_this_class(1:6)];
        y_val = [y_val; labels_this_class(7:8)];
        y_test = [y_test; labels_this_class(9:10)];
        X_train_no_pca = [X_train_no_pca; X_this_class_no_pca(1:6, :)];
        X_test_no_pca = [X_test_no_pca; X_this_class_no_pca(9:10, :)];
    end
    
    n_train=6*40;
    n_test = 2*40;
    n_val=2*40;
    
end

if split_dataset_again == 1
    if ~exist(path_save, 'dir')
        mkdir(path_save);
    end
    save(sprintf('%sX_train', path_save), 'X_train');
    save(sprintf('%sX_test', path_save), 'X_test');
    save(sprintf('%sy_train', path_save), 'y_train');
    save(sprintf('%sy_test', path_save), 'y_test');
else
    load(sprintf('%sX_train', path_save));
    load(sprintf('%sX_test', path_save));
    load(sprintf('%sy_train', path_save));
    load(sprintf('%sy_test', path_save));
end
% save(sprintf('%sy_train', path_save), 'y_train');
% save(sprintf('%sy_test', path_save), 'y_test');

if do_validation ==1
    if strcmp (dataset, 'wine') || strcmp (dataset, 'iris')
        n_val= round(0.5*length(y_test));
        n_test_new = n_test-n_val;
        val_indices = randperm(length(y_test), n_val);
        j_val=0;
        j_test=0;
        X_val = zeros(n_val , size(X_test,2));
        X_test_new = zeros(n_test_new , size(X_test,2));
        y_val = zeros(n_val, 1);
        y_test_new = zeros(n_test_new, 1);
        
        for j=1:size(X_test,1)
            if ismember(j,val_indices)
                j_val=j_val+1;
                X_val(j_val,:) = X_test(j, :);
                y_val(j_val) = y_test(j);
            else
                j_test=j_test+1;
                X_test_new(j_test, :)= X_test(j,:);
                y_test_new(j_test) = y_test(j);
            end
        end
        X_test=X_test_new;
        y_test = y_test_new;
    end
    if split_dataset_again == 1
        if ~exist(path_save, 'dir')
            mkdir(path_save)
        end
        save(sprintf('%sX_val', path_save), 'X_val')
        save(sprintf('%sX_test', path_save), 'X_test')
        save(sprintf('%sy_val', path_save), 'y_val')
        save(sprintf('%sy_test', path_save), 'y_test')
    else
        load(sprintf('%sX_val', path_save))
        load(sprintf('%sX_test', path_save))
        load(sprintf('%sy_val', path_save))
        load(sprintf('%sy_test', path_save))
    end
end
% save(sprintf('%sy_val', path_save), 'y_val')
% save(sprintf('%sy_test', path_save), 'y_test')

N_features = size(X_train, 2);

%% Optimization

%%%%%%%------ hierarchical method:
if hierarchical_mode == 1
    
    X_train_original = X_train;
    
    N_dimensions = size(X_train,2);
    std_ranges=zeros(N_dimensions,1);
    for dimension_index = 1:N_dimensions
        std_ranges(dimension_index) = std(X_train(:,dimension_index));
    end

    radius = 0.1*mean(std_ranges);
    % radius_step = 0.3*mean(std_ranges);
    n_hyperspheres = max(min(floor(0.01*size(X_train,1)),20),10);
    is_sample_seen_so_far = zeros(size(X_train,1),1);
    n_trial = 0;
    sampling_portion = 1;

    tic
    
    for iteration_index = 1:n_iteration
        str = sprintf('iteration index: %d', iteration_index);
        disp(str);

        std_ranges=zeros(N_dimensions,1);
        for dimension_index = 1:N_dimensions
            std_ranges(dimension_index) = std(X_train(:,dimension_index));
        end
        radius_step = 0.3*mean(std_ranges);
        %radius_step = 0.5*mean(std_ranges);  %--> not used in our datasets

        if iteration_index > 1
            radius = radius  + radius_step;
            n_hyperspheres = max(n_hyperspheres-ceil(0.2*n_hyperspheres),1);
            sampling_portion = max(sampling_portion - 0.05, 0.2);
        end

        for hypersphere_index = 1 : n_hyperspheres

            str = sprintf('iteration index: %d, hypersphere index: %d',iteration_index, hypersphere_index);
            disp(str);

            N_dimensions = size(X_train,2);
            center = zeros(1,N_dimensions);
            for dimension_index = 1:N_dimensions
                min_range_dim = min(X_train(:,dimension_index));
                max_range_dim = max(X_train(:,dimension_index));
                center(1,dimension_index) = min_range_dim + (max_range_dim-min_range_dim).*rand;
            end

            [X_train_subset,y_train_subset,~,triplet_indices_subset,is_sample_within_hypersphere, is_sample_selected_in_hypersphere] = hypersphere_sample_points(center,radius,sampling_portion, X_train,y_train,k,lambda_, method, method_index, do_sample_within_hypesphere);
            %[X_train_subset,y_train_subset,~,triplet_indices_subset,is_sample_selected_in_hypersphere] = bootstrap_sample_points(sampling_portion, X_train,y_train,k,lambda_, method, method_index);

            if ~isempty(triplet_indices_subset)

                is_sample_seen_so_far = is_sample_seen_so_far | is_sample_selected_in_hypersphere;

                if N_dimensions == 2
                    fig_ = figure('visible', 'off');
                    colors=['b', 'r', 'g', 'y', 'm', 'c', 'k'];
                    n_classes = length(unique(y_train));
                    pntColor = hsv(n_classes);
                    for class_index =1:n_classes
                        if n_classes <= 7
                            color_ = colors(class_index);
                        else
                            color_ = pntColor(class_index,:);
                        end
                        X_class = X_train(y_train==class_index,:);
                        plot(X_class(:,1), X_class(:,2),'color',color_,'linestyle','none','marker','o','MarkerFaceColor',color_);
                        hold on
                    end
                    for class_index =1:n_classes
                        if n_classes <= 7
                            color_ = colors(class_index);
                        else
                            color_ = pntColor(class_index,:);
                        end
                        X_class = X_train_subset(y_train_subset==class_index,:);
                        plot(X_class(:,1), X_class(:,2),'color',color_,'linestyle','none','marker','o','MarkerFaceColor',color_, 'MarkerSize', 16);
                        hold on
                    end
                    hold off
                    path_save_plot = sprintf('%splots/', path_save);
                    if ~exist(path_save_plot, 'dir')
                        mkdir(path_save_plot);
                    end
                    path_and_name = sprintf('%siteration%d_hypersphere%d_1(before).png', path_save_plot, iteration_index, hypersphere_index);
                    saveas(gcf, path_and_name)
                    close(fig_)
                end

                %----- optimization:
                [M, Xi] = Semidefinite_programing(X_train_subset, triplet_indices_subset, c);
                
                %----- Projecting data:
                if projectAllData_in_hierarchicalMode == 0
                    %----- Project the subset of data:
                    %M = ProjectOntoPositiveSemideinite(M);
                    %M = ProjectOntoPositiveSemideinite_epsilon(M, 0.001);
                    scatter_ = mean(std(X_train_subset, 1));
                    M = M + scatter_*eye(size(M));
                    [V,D] = eig(M);
                    [sorted_eigenValues, sorted_indices] = sort(diag(D), 'descend');
                    D_sorted = D(sorted_indices, sorted_indices);
                    V_sorted = V(:, sorted_indices);
                    D = D_sorted(:, :);
                    V = V_sorted(:, :);
                    %D(D<0) = 0;
                    L = V*(D^(0.5));

                    X_train_subset_projected = L'*X_train_subset';
                    X_train_subset_projected = X_train_subset_projected'; %row-wise
                    X_train(is_sample_selected_in_hypersphere==1,:)=X_train_subset_projected;
                else
                    %----- Project the whole data:
                    %M = ProjectOntoPositiveSemideinite(M);
                    %M = ProjectOntoPositiveSemideinite_epsilon(M, 0.001);
                    %scatter_ = mean(std(X_train, 1));
                    scatter_ = 1;
                    M = M + scatter_*eye(size(M));
                    [V,D] = eig(M);
                    [sorted_eigenValues, sorted_indices] = sort(diag(D), 'descend');
                    D_sorted = D(sorted_indices, sorted_indices);
                    V_sorted = V(:, sorted_indices);
                    D = D_sorted(:, :);
                    V = V_sorted(:, :);
                    %D(D<0) = 0;
                    L = V*(D^(0.5));

                    X_train = L'*X_train';
                    X_train = X_train'; %row-wise
                    X_train_subset_projected = L'*X_train_subset';
                    X_train_subset_projected = X_train_subset_projected'; %row-wise
                end
                
                %----- Log Results
                if log_results == 1
                    path_save_log = sprintf('%slog/', path_save);
                    if ~exist(path_save_log, 'dir')
                        mkdir(path_save_log);
                    end
                    save(sprintf('%sM_itr%d_hyper%d', path_save_log, iteration_index, hypersphere_index ), 'M');
                    save(sprintf('%sX_train_itr%d_hyper%d', path_save_log, iteration_index, hypersphere_index ), 'X_train');
                    save(sprintf('%sis_sample_selected_in_hypersphere_itr%d_hyper%d', path_save_log, iteration_index, hypersphere_index ), 'is_sample_selected_in_hypersphere');
                end

                if N_dimensions == 2
                    fig_ = figure('visible', 'off');
                    colors=['b', 'r', 'g', 'y', 'm', 'c', 'k'];
                    n_classes = length(unique(y_train));
                    pntColor = hsv(n_classes);
                    for class_index =1:n_classes
                        if n_classes <= 7
                            color_ = colors(class_index);
                        else
                            color_ = pntColor(class_index,:);
                        end
                        X_class = X_train(y_train==class_index,:);
                        plot(X_class(:,1), X_class(:,2),'color',color_,'linestyle','none','marker','o','MarkerFaceColor',color_);
                        hold on
                    end
                    for class_index =1:n_classes
                        if n_classes <= 7
                            color_ = colors(class_index);
                        else
                            color_ = pntColor(class_index,:);
                        end
                        X_class = X_train_subset_projected(y_train_subset==class_index,:);
                        plot(X_class(:,1), X_class(:,2),'color',color_,'linestyle','none','marker','o','MarkerFaceColor',color_, 'MarkerSize', 16);
                        hold on
                    end
                    hold off
                    path_and_name = sprintf('%siteration%d_hypersphere%d_2(after).png', path_save_plot, iteration_index, hypersphere_index);
                    saveas(gcf, path_and_name)
                    close(fig_)
                end

            end
            %pause(0.5);
        end
        if sum(is_sample_seen_so_far)~=size(X_train,1) && (iteration_index >= n_iteration) && n_trial <= 5
            iteration_index = iteration_index-1;
            n_trial= n_trial+1;
        end
    end
    time_ = toc;
    
    
    
    X_train = X_train_original;

%%%%%%%------ original method:
else

    y_ij = generate_y_ij(y_train);
    %N_train = size(X_train,1);

    if do_validation == 0
        
        tic
        
        triplet_indices = Calculate_eta_and_triplet_indices(X_train,y_train,k, n_train, y_ij, lambda_, method, method_index);
        
        time_triplet_generation = toc;

        tic

        [M, Xi] = Semidefinite_programing(X_train, triplet_indices, c);

        time_ = toc;

    else
        accuracy_val = zeros(length(k_list), length(c_list));
        y_pred_val = zeros(length(k_list), length(c_list), length(y_val));
        M_val = zeros(length(k_list), length(c_list), N_features, N_features);
        Xi_val = cell(length(k_list), length(c_list));
        for i = 1:length(k_list)
            k = k_list(i);
            str = sprintf('the k for validation is: %f', k );
            disp(str)
            triplet_indices = Calculate_eta_and_triplet_indices(X_train,y_train,k, n_train, y_ij, lambda_, method, method_index);
            for j = 1:length(c_list)
                c = c_list(j);
                str = sprintf('the c for validation is: %f' ,c );
                disp(str)
                [M, Xi] = Semidefinite_programing(X_train, triplet_indices, c);
                [y_pred, accuracy_] = knn_classification_Mahalanobis(X_train, y_train, X_val, y_val, k, M);
                accuracy_val(i,j) = accuracy_;
                y_pred_val(i,j,:) = y_pred;
                M_val(i,j,:,:) = M;
                Xi_val{i,j} = Xi;
                str = sprintf('the validation accuracy is %f: ',accuracy_val(i,j) );
                disp(str)
                save(sprintf('%saccuracy_val', path_save), 'accuracy_val');
                save(sprintf('%sy_pred_val', path_save), 'y_pred_val');
                save(sprintf('%sM_val', path_save), 'M_val');
                save(sprintf('%sXi_val', path_save), 'Xi_val');
            end
        end

        [best_k_index, best_c_index] = ind2sub([length(k_list), length(c_list)],find(accuracy_val == max(max(accuracy_val))));
        if length(best_k_index)>1
            best_k_index = best_k_index(1);
            best_c_index = best_c_index(1);
        end

        % train the model with best parameters:
        best_k = k_list(best_k_index);
        best_c = c_list(best_c_index);
        
        tic
        
        triplet_indices = Calculate_eta_and_triplet_indices(X_train,y_train,k, n_train, y_ij, lambda_, method, method_index);
        
        time_triplet_generation = toc;

        tic
        [M, Xi] = Semidefinite_programing(X_train, triplet_indices, best_c);
        time_ = toc;

    end
end

%% save results and plot:

save(sprintf('%sM', path_save), 'M');
save(sprintf('%sXi', path_save), 'Xi');

[y_pred_Mahalanobis, accuracy_test_Mahalanobis] = knn_classification_Mahalanobis(X_train, y_train, X_test, y_test, k, M);
[y_pred_Euclidean, accuracy_test_Euclidean] = knn_classification_Euclidean(X_train, y_train, X_test, y_test, k);
[y_pred_Energy, accuracy_Energy] = Energy_based_calssification(X_train, y_train, X_test, y_test, M, c);
[y_pred_SVM, accuracy_SVM] = multiclass_SVM_classification(X_train, y_train, X_test, y_test);


save(sprintf('%sy_pred_Mahalanobis', path_save), 'y_pred_Mahalanobis');
save(sprintf('%saccuracy_test_Mahalanobis', path_save), 'accuracy_test_Mahalanobis');
save(sprintf('%sy_pred_Euclidean', path_save), 'y_pred_Euclidean');
save(sprintf('%saccuracy_test_Euclidean', path_save), 'accuracy_test_Euclidean');
save(sprintf('%sy_pred_Energy', path_save), 'y_pred_Energy');
save(sprintf('%saccuracy_Energy', path_save), 'accuracy_Energy');
save(sprintf('%sy_pred_SVM', path_save), 'y_pred_SVM');
save(sprintf('%saccuracy_SVM', path_save), 'accuracy_SVM');
save(sprintf('%time', path_save), 'time_');


str = sprintf('the Mahalanobis test accuracy is %f: ',accuracy_test_Mahalanobis );
disp(str)
str = sprintf('the Euclidean test accuracy is %f: ',accuracy_test_Euclidean );
disp(str)
str = sprintf('the Energy test accuracy is %f: ',accuracy_Energy );
disp(str)
str = sprintf('the SVM test accuracy is %f: ',accuracy_SVM );
disp(str)

if hierarchical_mode ==0
    str = sprintf('the optimization time is %f: ', time_ );
    disp(str)
    str = sprintf('the triplet generation time is %f: ', time_triplet_generation );
    disp(str)
else
    str = sprintf('the total run time is %f: ', time_ );
    disp(str)
end

save(sprintf('%sworkspace', path_save));

plot_subspace(X_train, y_train, X_test, y_test, M, path_save);








