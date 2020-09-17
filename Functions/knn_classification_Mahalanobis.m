function [y_pred, accuracy_] = knn_classification_Mahalanobis(X_train, y_train, X_test, y_test, k, M)
M = project_on_semidefinete_cone(M);
M=M+0.0000001*eye(size(M));
Mdl = fitcknn(X_train,y_train,'NumNeighbors', k, 'Distance', 'mahalanobis', 'Cov', M);
y_pred = predict(Mdl,X_test);
accuracy_ = sum (y_test == y_pred)/length(y_test);

end

