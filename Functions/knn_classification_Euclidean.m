function [y_pred, accuracy_] = knn_classification_Euclidean(X_train, y_train, X_test, y_test, k)

Mdl = fitcknn(X_train,y_train,'NumNeighbors', k, 'Distance', 'euclidean');
y_pred = predict(Mdl,X_test);
accuracy_ = sum (y_test == y_pred)/length(y_test);

end

