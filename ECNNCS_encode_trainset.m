function ECNNCS_encode_trainset

clc
clear all
close all

patch_shift=200;
patch_size=200;
rows=2000;
cols=2000;
patch_count=1;
load('base-matrix-200-287.mat');
data_augment=0;%Need data_augment or not?
% create file for storing training patch paths and encoded signals
fid=fopen(['data/train-list.txt'],'w');

for patient_count=1:1:73
    
    struct=dir(['data/original/mitoses_ground_truth/',num2str(patient_count),'/*.csv']);
    image_num=size(struct,1);
    for image_count=1:1:image_num
        
        if image_count<10
            image_count=['0',num2str(image_count)];
        else
            image_count=num2str(image_count);
        end
        
        if ~exist(['data/original/mitoses_ground_truth/',num2str(patient_count),'/',image_count,'.csv']);
            continue
        end
        disp(['data/original/mitoses_ground_truth/',num2str(patient_count),'/',image_count,'.csv']);
        
        cor_Matrix = csvread(['data/original/mitoses_ground_truth/',num2str(patient_count),'/',image_count,'.csv']);
        mitos_num=size(cor_Matrix,1);
        MatData=zeros(rows,cols);
        row=cor_Matrix(:,1);
        col=cor_Matrix(:,2);
        for i=1:1:mitos_num
            MatData(row(i),col(i))=1;
        end
        
        if patient_count<15
            image=imread(['data/original/mitoses_image_data_part_1/',num2str(patient_count),'/',image_count,'.tif']);
        elseif patient_count<34
            image=imread(['data/original/mitoses_image_data_part_2/',num2str(patient_count),'/',image_count,'.tif']);
        else
            image=imread(['data/original/mitoses_image_data_part_3/',num2str(patient_count),'/',image_count,'.tif']);
        end
        
        %Partition images to patches
        mkdir('data/train_patches');
        for m=1:patch_shift:rows-patch_size+1,
            for n=1:patch_shift:cols-patch_size+1,
                blkMat = MatData(m:m+patch_size-1,n:n+patch_size-1);
                patch = image(m:m+patch_size-1,n:n+patch_size-1,:);
                %plot([n; n+patch_size; n+patch_size; n; n], [m; m; m+patch_size; m+patch_size; m],'g');
                Enc_signal=radon_rp_encode(blkMat,G);
                % save training patch and its encoded signal
                image_path=['data/train_patches/',num2str(patient_count),'-',num2str(image_count),'-',num2str(patch_count),'.jpg'];
                imwrite(patch,image_path,'jpg');
                write_train_list(image_path,Enc_signal,fid,patch_count);
                fprintf(fid,[image_path,' ']);
                fprintf(fid,num2str(Enc_signal'));
                fprintf(fid,'\r\n');
                patch_count=patch_count+1;
                
                if data_augment==1
                    rot_patch=patch;
                    rot_blkMat=blkMat;
                    for times=1:1:3%rotate each patch iteratively
                        rot_patch=imrotate(rot_patch,90);%Left rotate 90 degree
                        rot_blkMat=rot90(rot_blkMat);%Left rotate 90 degree
                        Enc_signal=radon_rp_encode(rot_blkMat,G);
                        image_path=['data/train_patches/',num2str(patient_count),'-',num2str(image_count),'-',...
                            num2str(patch_count),'-rotate',num2str(times),'.jpg'];
                        imwrite(rot_patch,image_path,'jpg');
                        write_train_list(image_path,Enc_signal,fid,patch_count);
                        patch_count=patch_count+1;
                    end
                end
                
            end
        end
    end
end
fclose(fid);
end