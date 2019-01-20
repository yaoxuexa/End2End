function write_train_list(image_path,Enc_signal,fid,patch_count)

fprintf(fid,'%s\t',num2str(patch_count));
%fprintf(fid,[num2str(Enc_signal'),' ']);
label_width=length(Enc_signal);
for i=1:1:label_width
    label=Enc_signal(i);
    fprintf(fid,'%s\t',num2str(label));
end
fprintf(fid,'%s\r\n',image_path);

end