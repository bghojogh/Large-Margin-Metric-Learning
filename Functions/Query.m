function Query(X_train,X_test,X_train_no_pca, image_size,M ,X_test_no_pca)

[mIdx,~] = knnsearch(X_test,X_train,'K',10, 'Distance', 'mahalanobis', 'Cov', M);
%[mIdx,mD] = knnsearch(X_test,X_train,'K',10);
for test_index = 1:size(X_test,1)
    for neighbour_index = 1:size(mIdx, 2)
        train_index = mIdx(test_index, neighbour_index);
        path_image = sprintf('./query/%d/', test_index);
        path_image_ = sprintf('%s%d.jpeg', path_image, neighbour_index);
        if ~exist(path_image, 'dir')
            mkdir(path_image);
        end
        imwrite(reshape(X_train_no_pca(train_index,:)./255, image_size), path_image_,'JPEG');
    end
    path_image_test = sprintf('%stest.jpeg', path_image);
    imwrite(reshape(X_test_no_pca(test_index,:)./255, image_size), path_image_test,'JPEG');
end

end

