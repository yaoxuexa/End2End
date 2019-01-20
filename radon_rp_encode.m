function Enc_signal=radon_rp_encode(blkMat,G)
    % uniformly spaced lines; higher value gets better reconstruction
    r=40;
    % radon transform to compress cell locations
    theta = linspace(0,179,r); 
    radon_matrix = radon(blkMat,theta);
    Enc_signal=radon_matrix(:);
    % notice that each column of R is sparse
    % apply random projection
    B = G*radon_matrix;
    Enc_signal=[Enc_signal;B(:)];
end