function ECNNCS_test

clc
clear all
close all

patch_shift=200;
patch_size=200;
rows=5657;
cols=5657;

for patient_count=4:1:43
    
    if patient_count<10
        patient_count=['0',num2str(patient_count)];
    else
        patient_count=num2str(patient_count);
    end
    
    cor_Matrix = csvread(['AMIDA16-ECNNCS/',patient_count,'/01.csv']);
    mitos_num=size(cor_Matrix,1);
    MatData=zeros(rows,cols);
    row=cor_Matrix(:,1);
    col=cor_Matrix(:,2);
    for i=1:1:mitos_num
        MatData(row(i),col(i))=1;
    end
    
    image=imread(['data/original/mitoses-test-image-data/',patient_count,'/01.tif']);
    figure(1),imshow(image),hold on;
    for m=1:patch_shift:rows-patch_size+1,
        for n=1:patch_shift:cols-patch_size+1,
            plot([n; n+patch_size; n+patch_size; n; n], [m; m; m+patch_size; m+patch_size; m],'g');
            
            blkMat = MatData(m:m+patch_size-1,n:n+patch_size-1);
            [x,y]=find(blkMat==1);
            if ~isempty(x)
                patch = image(m:m+patch_size-1,n:n+patch_size-1,:);
                h=figure(2),imshow(patch),hold on,plot(y,x,'g+','LineWidth',1),hold off;
                pause(3),close(h);
            end
        end
    end
    hold off;
end
end