function [A] = ProjectOntoPositiveSemideinite_epsilon(A, epsilon)

    [V, D] = eig(A);
    eigen_Values = diag(D);
    eigen_Values(eigen_Values<epsilon) = epsilon;
    D = diag(eigen_Values);
    A = V * D * transpose(V);

end