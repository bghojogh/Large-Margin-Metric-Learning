function ghost_faces = ghost_image(M, image_size, PCA_projection_directions)

[V,D] = eig(M);
[sorted_eigenValues, sorted_indices] = sort(diag(D), 'descend');
D_sorted = D(sorted_indices, sorted_indices);
V_sorted = V(:, sorted_indices);

D_complete = D_sorted(:,:);
V_complete = V_sorted(:,:);
L_complete = V_complete*(D_complete^(0.5));
L_complete_reconstructed = PCA_projection_directions * L_complete;
n_ghost_faces = size(L_complete_reconstructed, 2);
ghost_faces=cell(n_ghost_faces,1);


for ghost_face_index = 1:n_ghost_faces
    ghost_face = L_complete_reconstructed(:,ghost_face_index);
    ghost_face = ghost_face - min(ghost_face);
    ghost_face = ghost_face/max(ghost_face);
    ghost_face = ghost_face*255;
    ghost_faces{ghost_face_index}(:,:) = reshape(ghost_face, image_size);
    figure
    imshow(ghost_faces{ghost_face_index}(:,:))
end


end

