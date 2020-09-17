function [y_pred, accuracy_] = multiclass_SVM_classification(X_train, y_train, X_test, y_test)

classes = unique(y_train);
n_classes = length(classes);
SVMModels = cell(n_classes, 1);
rng(1); % For reproducibility

for j = 1:numel(classes)
    indx = (y_train==classes(j)); % Create binary classes for each classifier
    SVMModels{j} = fitcsvm(X_train,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','rbf');
end

Scores = zeros(length(y_test),numel(classes));

for j = 1:numel(classes)
    [~,score] = predict(SVMModels{j},X_test);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,maxScore_index] = max(Scores,[],2);
y_pred = maxScore_index;
accuracy_ = sum (y_test == y_pred)/length(y_test);

end

