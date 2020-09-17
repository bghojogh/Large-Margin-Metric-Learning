function plot_subspace(X_train, y_train, X_test, y_test, M, path_save)

% M = M + 1*eye(size(M));
[V,D] = eig(M);
[sorted_eigenValues, sorted_indices] = sort(diag(D), 'descend');
D_sorted = D(sorted_indices, sorted_indices);
V_sorted = V(:, sorted_indices);
D = D_sorted(1:2, 1:2);
V = V_sorted(:, 1:2);
L = V*(D^(0.5));

X_train_projected = L'*X_train';
X_test_projected = L'*X_test';
X_train_projected = X_train_projected'; %row-wise
X_test_projected = X_test_projected';    %row-wise


% figure;
fig_ = figure('visible', 'off');
colors=['b', 'r', 'g', 'y', 'm', 'c', 'k'];
n_classes = length(unique(y_train));
pntColor = hsv(n_classes);

for class_index = 1:n_classes
    X_train_projected_masked = X_train_projected(y_train==class_index, :);
    X_test_projected_masked = X_test_projected(y_test==class_index, :);
    legend_txt1 = sprintf('Class %d, train set', class_index);
    legend_txt2 = sprintf('Class %d, test set', class_index);
    if n_classes <= 7
        color_ = colors(class_index);
    else
        color_ = pntColor(class_index,:);
    end
    plot(X_train_projected_masked(:,1), X_train_projected_masked(:,2), 'o', 'color', color_, 'MarkerFaceColor',color_, 'DisplayName', legend_txt1);
    hold on
    plot(X_test_projected_masked(:,1), X_test_projected_masked(:,2), '*', 'color', color_, 'MarkerFaceColor',color_, 'DisplayName', legend_txt2);
end

%legend show
xlabel('Projection Direction 1');
ylabel('Projection Direction 2');

hold off
path_and_name = sprintf('%sFinal_embedding.png', path_save);
saveas(gcf, path_and_name)
close(fig_)


end

