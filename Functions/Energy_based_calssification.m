function [y_pred, accuracy_] = Energy_based_calssification(X_train, y_train, X_test, y_test, M, c)

n_train = length(y_train);
y_pred = zeros(length(y_test), 1);
for test_sample_index=1:length(y_test)
    loss_min=inf;
    for class_index = 1: length(unique(y_train))
        for j=1:n_train
            if y_train(j) == class_index
                for l=1:n_train
                    if y_train(l) ~= class_index
                        term1 = (X_train(j,:)'-X_test(test_sample_index,:)').' ...
                            *M*(X_train(j,:)'-X_test(test_sample_index,:)');
                        
                        hinge_loss =max(1+(X_train(j,:)'-X_test(test_sample_index,:)').' ...
                            *M*(X_train(j,:)'-X_test(test_sample_index,:)') ...
                            -(X_train(l,:)'-X_test(test_sample_index,:)').' ...
                            *M*(X_train(l,:)'-X_test(test_sample_index,:)'),0);
                        
                        loss = term1 + c*hinge_loss;
                        if loss <= loss_min
                            loss_min = loss;
                            y_pred(test_sample_index) = class_index;
                        end
                    end
                end
            end
        end
    end
end

accuracy_ = sum (y_test == y_pred)/length(y_test);

end

