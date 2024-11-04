% data setting
clear all;
startup_bbci_toolbox
BTB.DataDir = 'D:\Memory-Data\EEG Data\Test_Data'; % data directory
BTB.filename = 'S4'; % subject file
BTB.RawDir = fullfile(BTB.DataDir,BTB.filename);
labelDir='D:\Memory-Data\Encoding-Labels';

%Read Label
va=open(fullfile(labelDir,strcat(BTB.filename,'.mat')));
label=va.label;

%% Load data
[cnt, mrk, mnt]= file_readBV(BTB.RawDir, 'Fs', 1000);

%% Preprocessing
[cnt, mrk]= proc_resample(cnt, 250, 'mrk', mrk,'N',30);

%band pass filter
band=[3 40];
[b,a]= butter(5,band/cnt.fs*2);
cnt=proc_filtfilt(cnt,b,a);

%% Select channel
cnt=proc_selectChannels(cnt,'Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','CP5','CP1','CP2','CP6','P7',...
         'P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10','AF7','AF3','AF4','AF8','F5','F1','F2','F6','FT9','FT7','FC3','FC4','FT8','FT10',...
         'C5','C1','C2','C6','TP7','CP3','CPz','CP4','TP8','P5','P1','P2','P6','PO7', 'PO3','POz ','PO4',  'PO8');
%cnt=proc_selectChannels(cnt,'FC1','FC2','C3','Cz','C4','C1','C2','F7','F3','FC5','T7','CP5','P7','F5','FT9','FT7','FC3','C5','TP7');
mnt_eeg= mnt_setElectrodePositions(cnt.clab);
mnt_eeg=  mnt_setGrid(mnt_eeg);    
% %% Band of interest beta
% bands=[13 30];
% [filt_b, filt_a] = butters(5,bands/cnt.fs*2);
% cnt = proc_filterbank(cnt,filt_b,filt_a);
%% Markers setting
disp_cross= [-1000 0];  %1s before stimulation
disp_stimulus=[0 1000];
%trig= {1; 'Cross'};    %Cross Trig
trig= {2; 'Stimulus'};    %Stimulus Trig
mrk_cross= mrk_defineClasses(mrk, trig);
mrk_cross.orig=mrk;    

%% Epoch Cross 
%epo=proc_segmentation(cnt,mrk_cross, disp_cross); %Cross Epoch

%% Epoch  Stimulus
epo=proc_segmentation(cnt,mrk_cross, disp_stimulus); %Stimulus Epoch

epo.y=label';    %Update real label
epo.y(2,:)=~label';
epo=rmfield(epo,'className');
epo.className={'Remember','Forgotten'};

%% Reselect trials
%for s=1:5
%rng(s)
 r_n=sum(epo.y(1,:));
 f_n=250-r_n;

    
new_epo=epo;  
%% Classification_CSP
% Select discriminative time intervals

[csp_fv,csp_w,csp_eig]=proc_multicsp(new_epo,3);

proc=struct('memo','csp_w');
    
    proc.train= ['[fv,csp_w]=  proc_multicsp(fv, 3); ' ...
        'fv= proc_variance(fv); ' ...
        'fv= proc_logarithm(fv);'];
                
proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ','fv= proc_variance(fv); ' ,'fv= proc_logarithm(fv);'];

[C_eeg, loss_eeg_std, out_eeg.out, memo] = xvalidation(new_epo,'RLDAshrink','proc',proc, 'kfold', 10);

confu_matrix=confusionMatrix(new_epo.y,out_eeg.out);
acc= (1-C_eeg)*100
loss_eeg_std=loss_eeg_std*100
% exp_acc= ((r_n*(confu_matrix(1,1)+ confu_matrix(2,1))/(250*10))+(f_n*(confu_matrix(1,2)+ confu_matrix(2,2))/(250*10)))/250;
% kappa=(acc/100-exp_acc)/(1-exp_acc)
%end 
opt.colAx=[-0.2 0.2]
figure(2)
plotCSPatterns(csp_fv, mnt_eeg, csp_w,csp_fv.y);

%fastAUC(logical(new_epo.y),out_eeg.out,1)
