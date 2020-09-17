function eta = generate_eta_friend(X_train, y_train, k, get_nearest)

%X_train is row wise
N_train = size(X_train,1);

% the first input is the row wise data set = X_train 
% the second input is points we want to find the k nearest neighbour for
% input 1 = input 2 so we can find the k nearest neighbour for each data of
% our data set
% mIdx = is the k indices = it is the indices of k nearest neighbours from
% data sets
% mD is the disttances of the k nearest neighbour

[mIdx,~] = knnsearch(X_train,X_train,'K',N_train);

%N_train is the sample size of the train set
eta = zeros(N_train, N_train);
for i=1:N_train
    counter=0;
    if get_nearest   % nearest neighbors from friends (same class):
        for j=1:N_train
            index_=mIdx(i,j);
            if index_~= i && y_train(i) == y_train(index_)
                eta(i, index_) = 1;
                counter = counter+1;
            end
            if counter >= k
                break
            end
        end
    else  % furthest neighbors from friends (same class):
        for j=N_train:-1:1
            index_=mIdx(i,j);
            if index_~= i && y_train(i) == y_train(index_)
                eta(i, index_) = 1;
                counter = counter+1;
            end
            if counter >= k
                break
            end
        end
    end
end
end

