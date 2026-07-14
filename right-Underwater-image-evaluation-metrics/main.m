clc;
clear;
close all;


%% 总文件夹
root_folder = ...
'E:\A我的孩子们\1.水下图像增强\3.结果图';



%% 获取方法

method_dirs = dir(root_folder);

method_dirs = method_dirs([method_dirs.isdir]);

method_dirs = method_dirs(~ismember({method_dirs.name},{'.','..'}));



Results = {};

count = 1;



for m = 1:length(method_dirs)


    method_name = method_dirs(m).name;

    method_path = fullfile(root_folder,method_name);


    fprintf('\n========== %s ==========\n',method_name);



    %% 获取数据集

    dataset_dirs = dir(method_path);

    dataset_dirs = dataset_dirs([dataset_dirs.isdir]);

    dataset_dirs = dataset_dirs(~ismember({dataset_dirs.name},{'.','..'}));



    for d = 1:length(dataset_dirs)


        dataset_name = dataset_dirs(d).name;

        dataset_path = fullfile(method_path,dataset_name);



        fprintf('\n数据集: %s\n',dataset_name);



        %% 搜索图片

        img_files = dir(fullfile(dataset_path,'*.jpg'));

        if isempty(img_files)
            img_files = dir(fullfile(dataset_path,'*.png'));
        end


        if isempty(img_files)

            fprintf('没有图像，跳过\n');
            continue;

        end



        total_uciqe = 0;
        total_uiqm = 0;
        total_uicm = 0;
        total_uism = 0;


        num = 0;



        for i = 1:length(img_files)


            img_path = fullfile(dataset_path,...
                img_files(i).name);


            img = imread(img_path);


            % 删除alpha通道
            if size(img,3)>3
                img = img(:,:,1:3);
            end



            %% ==========================
            % UCIQE
            %% ==========================

            uciqe_value = UCIQE(img);



            %% ==========================
            % UIQM
            %% ==========================

            uiqm_value = UIQM(img);



            %% ==========================
            % UICM
            %% ==========================

            [~,~,~,~,uicm_value] = UICM(img);



            %% ==========================
            % UISM
            %% ==========================

            uism_value = UISM(img);



            total_uciqe = total_uciqe + uciqe_value;

            total_uiqm = total_uiqm + uiqm_value;

            total_uicm = total_uicm + uicm_value;

            total_uism = total_uism + uism_value;


            num=num+1;


        end



        %% 平均指标


        avg_uciqe = total_uciqe / num;

        avg_uiqm = total_uiqm / num;

        avg_uicm = total_uicm / num;

        avg_uism = total_uism / num;



        fprintf('UCIQE = %.4f\n',avg_uciqe);
        fprintf('UIQM  = %.4f\n',avg_uiqm);
        fprintf('UICM  = %.4f\n',avg_uicm);
        fprintf('UISM  = %.4f\n',avg_uism);



        Results(count,:) = {
            method_name,...
            dataset_name,...
            avg_uciqe,...
            avg_uiqm,...
            avg_uicm,...
            avg_uism
            };


        count=count+1;



    end

end




%% 保存Excel

T = cell2table(Results,...
    'VariableNames',...
    {'Method',...
     'Dataset',...
     'UCIQE',...
     'UIQM',...
     'UICM',...
     'UISM'});


save_file = fullfile(root_folder,...
    'Underwater_NoReference_Metrics.xlsx');


writetable(T,save_file);


fprintf('\n================================\n');
fprintf('全部完成\n');
fprintf('结果保存:\n%s\n',save_file);
fprintf('================================\n');