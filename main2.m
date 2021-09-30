clear all;

% define the random number seed for repeatable results
rng(1,'twister');

%% Load Speech Commands Data 

% define the folder for the speech dataset
datasetFolder = 'speechDataReduced';

% Create an audioDatastore that points to the data set
ads = audioDatastore(datasetFolder, ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames');

%% Choose Words to Recognize

% define a subset of command words to recognise from the full data set
commands = categorical(["yes","no","up","down","left","right","on","off","stop","dog","cat","bird","wow","five","go"]);

% define an index into command words and unknown words
isCommand = ismember(ads.Labels,commands);
isUnknown = ~ismember(ads.Labels,[commands,"_background_noise_"]);

% specify the fraction of unknown words to include - labeling words that
% are not commands as |unknown| creates a group of words that approximates
% the distribution of all words other than the commands. The network uses
% this group to learn the difference between commands and all other words.
includeFraction = 0.1;
mask = rand(numel(ads.Labels),1) < includeFraction;
isUnknown = isUnknown & mask;
ads.Labels(isUnknown) = categorical("unknown");

% create a new data store of only command words and unknown words 
adsSubset = subset(ads,isCommand|isUnknown);

% count the number of instances of each word
countEachLabel(adsSubset)

%% Define Training, Validation, and Test data Sets

% define split proportion for training, validation and test data
p1 = 0.8; % training data proportion
p2 = 0.2; % validation data proportion
[adsTrain,adsValidation] = splitEachLabel(adsSubset,p1);

% reduce the dataset to speed up training 
numUniqueLabels = numel(unique(adsTrain.Labels));
nReduce = 2; % Reduce the dataset by a factor of nReduce
adsTrain = splitEachLabel(adsTrain,round(numel(adsTrain.Files) / numUniqueLabels / nReduce));
adsValidation = splitEachLabel(adsValidation,round(numel(adsValidation.Files) / numUniqueLabels / nReduce));

%% define object for computing auditory spectrograms from audio data

% spectrogram parameters
fs = 16e3;             % sample rate of the data set
segmentDuration = 1;   % duration of each speech clip (in seconds)
frameDuration = 0.025; % duration of each frame for spectrum calculation
hopDuration = 0.010;   % time step between each spectrum

segmentSamples = round(segmentDuration*fs); % number of segment samples
frameSamples = round(frameDuration*fs);     % number of frame samples
hopSamples = round(hopDuration*fs);         % number of hop samples
overlapSamples = frameSamples - hopSamples; % number of overlap samples

FFTLength = 512;  % number of points in the FFT
numBands = 50;    % number of filters in the auditory spectrogram

% extract audio features using spectrogram - specifically bark spectrum
afe = audioFeatureExtractor( ...
    'SampleRate',fs, ...
    'FFTLength',FFTLength, ...
    'Window',hann(frameSamples,'periodic'), ...
    'OverlapLength',overlapSamples, ...
    'barkSpectrum',true);
setExtractorParams(afe,'barkSpectrum','NumBands',numBands);

%% Process a file from the dataset to get denormalization factor

% apply zero-padding to the audio signal so they are a consistent length of 1 sec
x = read(adsTrain);
numSamples = size(x,1);
numToPadFront = floor( (segmentSamples - numSamples)/2 );
numToPadBack = ceil( (segmentSamples - numSamples)/2 );
xPadded = [zeros(numToPadFront,1,'like',x);x;zeros(numToPadBack,1,'like',x)];

% extract audio features - the output is a Bark spectrum with time across rows
features = extract(afe,xPadded);
[numHops,numFeatures] = size(features);

% determine the denormalization factor to apply for each signal
unNorm = 2/(sum(afe.Window)^2);

%% Feature extraction: read data file, zero pad, then apply spectrogram methods

% Training data: read from datastore, zero-pad, extract spectrogram features
subds = partition(adsTrain,1,1);
XTrain = zeros(numHops,numBands,1,numel(subds.Files));
for idx = 1:numel(subds.Files)
    x = read(subds);
    xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
    XTrain(:,:,:,idx) = extract(afe,xPadded);
end
XTrainC{1} = XTrain;
XTrain = cat(4,XTrainC{:});

% extract parameters from training data
[numHops,numBands,numChannels,numSpec] = size(XTrain);

% Scale the features by the window power and then take the log (with small offset)
XTrain = XTrain/unNorm;
epsil = 1e-6;
XTrain = log10(XTrain + epsil);

% Validation data: read from datastore, zero-pad, extract spectrogram features
subds = partition(adsValidation,1,1);
XValidation = zeros(numHops,numBands,1,numel(subds.Files));
for idx = 1:numel(subds.Files)
    x = read(subds);
    xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
    XValidation(:,:,:,idx) = extract(afe,xPadded);
end
XValidationC{1} = XValidation;
XValidation = cat(4,XValidationC{:});
XValidation = XValidation/unNorm;
XValidation = log10(XValidation + epsil);

% Isolate the train and validation labels. Remove empty categories.
YTrain = removecats(adsTrain.Labels);
YValidation = removecats(adsValidation.Labels);

% Visualize Data - plot waveforms and play clips for a few examples
specMin = min(XTrain,[],'all');
specMax = max(XTrain,[],'all');
idx = randperm(numel(adsTrain.Files),3);
figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
for i = 1:3
    [x,fs] = audioread(adsTrain.Files{idx(i)});
    subplot(2,3,i)
    plot(x)
    axis tight
    title(string(adsTrain.Labels(idx(i))))
    
    subplot(2,3,i+3)
    spect = (XTrain(:,:,1,idx(i))');
    pcolor(spect)
    caxis([specMin specMax])
    shading flat
    
    sound(x,fs)
    pause(2)
end

%% Add Background Noise Data
% The network must be able not only to recognize different spoken words but
% also to detect if the input contains silence or background noise.
%
% Use the audio files in the |_background_noise|_ folder to create samples
% of one-second clips of background noise. Create an equal number of
% background clips from each background noise file. You can also create
% your own recordings of background noise and add them to the
% |_background_noise|_ folder. Before calculating the spectrograms, the
% function rescales each audio clip with a factor sampled from a
% log-uniform distribution in the range given by |volumeRange|.

% Use the audio files in the |_background_noise|_ folder to create samples
% of background noise
adsBkg = subset(ads,ads.Labels=="_background_noise_");
numBkgClips = 400;


numBkgFiles = numel(adsBkg.Files);
numClipsPerFile = histcounts(1:numBkgClips,linspace(1,numBkgClips,numBkgFiles+1));
Xbkg = zeros(size(XTrain,1),size(XTrain,2),1,numBkgClips,'single');
bkgAll = readall(adsBkg);
ind = 1;

% rescale audio clip with value sampled from volumeRange 
volumeRange = log10([1e-4,1]);
for count = 1:numBkgFiles
    bkg = bkgAll{count};
    idxStart = randi(numel(bkg)-fs,numClipsPerFile(count),1);
    idxEnd = idxStart+fs-1;
    gain = 10.^((volumeRange(2)-volumeRange(1))*rand(numClipsPerFile(count),1) + volumeRange(1));
    for j = 1:numClipsPerFile(count)
        
        x = bkg(idxStart(j):idxEnd(j))*gain(j);
        
        x = max(min(x,1),-1);
        
        Xbkg(:,:,:,ind) = extract(afe,x);
        
        if mod(ind,1000)==0
            disp("Processed " + string(ind) + " background clips out of " + string(numBkgClips))
        end
        ind = ind + 1;
    end
end
Xbkg = Xbkg/unNorm;
Xbkg = log10(Xbkg + epsil);


% Split spectrograms of background noise between training and validation data
numTrainBkg = floor(0.85*numBkgClips);
numValidationBkg = floor(0.15*numBkgClips);
XTrain(:,:,:,end+1:end+numTrainBkg) = Xbkg(:,:,:,1:numTrainBkg);
YTrain(end+1:end+numTrainBkg) = "background";
XValidation(:,:,:,end+1:end+numValidationBkg) = Xbkg(:,:,:,numTrainBkg+1:end);
YValidation(end+1:end+numValidationBkg) = "background";

% Plot the distribution of the different class labels in the training and
% validation sets.

figure('Units','normalized','Position',[0.2 0.2 0.5 0.5])
subplot(2,1,1)
histogram(YTrain)
title("Training Label Distribution")
subplot(2,1,2)
histogram(YValidation)
title("Validation Label Distribution")

%% Define Deep Convolutional Neural Network Architecture
% Create a simple network architecture as an array of layers. 
% -Use convolutional and batch normalization layers
% -downsample the feature maps "spatially" (that is, in time and frequency) 
% using max pooling layers. 
% -Add a final max pooling layer that pools the input feature map
% globally over time. This enforces (approximate) time-translation
% invariance in the input spectrograms, allowing the network to perform the
% same classification independent of the exact position of the speech in
% time. 
% -Global pooling also significantly reduces the number of parameters
% in the final fully connected layer. 
% -To reduce the possibility of the
% network memorizing specific features of the training data, add a small
% amount of dropout to the input to the last fully connected layer.

% number of classes
numClasses = numel(categories(YTrain));

% time pooling size
timePoolSize = ceil(numHops/8);

% network parameters
dropoutProb = 0.2; % dropout probability
numF = 5;         % number of filters

lgraph = layerGraph();

tempLayers = [
    imageInputLayer([98 50 1],"Name","data")
    convolution2dLayer([7 7],64,"Name","conv1-7x7_s2","BiasLearnRateFactor",2,"Padding",[3 3 3 3],"Stride",[2 2])
    reluLayer("Name","conv1-relu_7x7")
    maxPooling2dLayer([3 3],"Name","pool1-3x3_s2","Padding",[0 1 0 1],"Stride",[2 2])
    crossChannelNormalizationLayer(5,"Name","pool1-norm1","K",1)
    convolution2dLayer([1 1],64,"Name","conv2-3x3_reduce","BiasLearnRateFactor",2)
    reluLayer("Name","conv2-relu_3x3_reduce")
    convolution2dLayer([3 3],192,"Name","conv2-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","conv2-relu_3x3")
    crossChannelNormalizationLayer(5,"Name","conv2-norm2","K",1)
    maxPooling2dLayer([3 3],"Name","pool2-3x3_s2","Padding",[0 1 0 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_3a-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],32,"Name","inception_3a-pool_proj","BiasLearnRateFactor",2)
    reluLayer("Name","inception_3a-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","inception_3a-1x1","BiasLearnRateFactor",2)
    reluLayer("Name","inception_3a-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],16,"Name","inception_3a-5x5_reduce","BiasLearnRateFactor",2)
    reluLayer("Name","inception_3a-relu_5x5_reduce")
    convolution2dLayer([5 5],32,"Name","inception_3a-5x5","BiasLearnRateFactor",2,"Padding",[2 2 2 2])
    reluLayer("Name","inception_3a-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],96,"Name","inception_3a-3x3_reduce","BiasLearnRateFactor",2)
    reluLayer("Name","inception_3a-relu_3x3_reduce")
    convolution2dLayer([3 3],128,"Name","inception_3a-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","inception_3a-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(4,"Name","inception_3a-output");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],32,"Name","inception_3b-5x5_reduce","BiasLearnRateFactor",2)
    reluLayer("Name","inception_3b-relu_5x5_reduce")
    convolution2dLayer([5 5],96,"Name","inception_3b-5x5","BiasLearnRateFactor",2,"Padding",[2 2 2 2])
    reluLayer("Name","inception_3b-relu_5x5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([3 3],"Name","inception_3b-pool","Padding",[1 1 1 1])
    convolution2dLayer([1 1],64,"Name","inception_3b-pool_proj","BiasLearnRateFactor",2)
    reluLayer("Name","inception_3b-relu_pool_proj")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","inception_3b-3x3_reduce","BiasLearnRateFactor",2)
    reluLayer("Name","inception_3b-relu_3x3_reduce")
    convolution2dLayer([3 3],192,"Name","inception_3b-3x3","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","inception_3b-relu_3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","inception_3b-1x1","BiasLearnRateFactor",2)
    reluLayer("Name","inception_3b-relu_1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","inception_3b-output")
    globalAveragePooling2dLayer("Name","pool5-7x7_s1")
    dropoutLayer(0.4,"Name","pool5-drop_7x7_s1")
    fullyConnectedLayer(17,"Name","loss3-classifier","BiasLearnRateFactor",2)
    softmaxLayer("Name","prob")
    classificationLayer("Name","output")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-pool");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-1x1");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-5x5_reduce");
lgraph = connectLayers(lgraph,"pool2-3x3_s2","inception_3a-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_3a-relu_3x3","inception_3a-output/in2");
lgraph = connectLayers(lgraph,"inception_3a-relu_pool_proj","inception_3a-output/in4");
lgraph = connectLayers(lgraph,"inception_3a-relu_1x1","inception_3a-output/in1");
lgraph = connectLayers(lgraph,"inception_3a-relu_5x5","inception_3a-output/in3");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-5x5_reduce");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-pool");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-3x3_reduce");
lgraph = connectLayers(lgraph,"inception_3a-output","inception_3b-1x1");
lgraph = connectLayers(lgraph,"inception_3b-relu_3x3","inception_3b-output/in2");
lgraph = connectLayers(lgraph,"inception_3b-relu_pool_proj","inception_3b-output/in4");
lgraph = connectLayers(lgraph,"inception_3b-relu_5x5","inception_3b-output/in3");
lgraph = connectLayers(lgraph,"inception_3b-relu_1x1","inception_3b-output/in1");
analyzeNetwork(lgraph)
plot(lgraph);
%% Training

options = trainingOptions('adam', ...
    'MaxEpochs',25, ...
    'MiniBatchSize',128, ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{XValidation,YValidation});

% Train the network
trainedNet = trainNetwork(XTrain,YTrain,lgraph,options);

%% Evaluate Trained Network
% Calculate the accuracy on the training, validation and test data
YValPred = classify(trainedNet,XValidation);
validationError = mean(YValPred ~= YValidation);
YTrainPred = classify(trainedNet,XTrain);
trainError = mean(YTrainPred ~= YTrain);
disp("Training error: " + trainError*100 + "%")
disp("Validation error: " + validationError*100 + "%")

% Plot the confusion matrix. Display the precision and recall for each class
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
cm = confusionchart(YValidation,YValPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
sortClasses(cm, [commands,"unknown","background"])

% Compute the total size of the network in kilobytes and test prediction speed 
info = whos('trainedNet');
disp("Network size: " + info.bytes/1024 + " kB")
for i = 1:100
    x = randn([numHops,numBands]);
    tic
    [YPredicted,probs] = classify(trainedNet,x,"ExecutionEnvironment",'cpu');
    time(i) = toc;
end
disp("Single-image prediction time on CPU: " + mean(time(11:end))*1000 + " ms")

% the end



