% Load training and test data using |imageDatastore|.
syntheticDir   = fullfile(toolboxdir('vision'), 'visiondata','digits','synthetic');
handwrittenDir = fullfile(toolboxdir('vision'), 'visiondata','digits','handwritten');

% |imageDatastore| recursively scans the directory tree containing the
% images. Folder names are automatically used as labels for each image.
trainingSet = imageDatastore(syntheticDir,   'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet     = imageDatastore(handwrittenDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');


% TO TRY
figure;

subplot(2,3,1);
imshow(trainingSet.Files{11});

subplot(2,3,2);
imshow(trainingSet.Files{31});

subplot(2,3,3);
imshow(trainingSet.Files{81});

subplot(2,3,4);
imshow(testSet.Files{6});

subplot(2,3,5);
imshow(testSet.Files{16});

subplot(2,3,6);
imshow(testSet.Files{41});

% Show pre-processing results
exTestImage = readimage(testSet,6);
processedImage = imbinarize(rgb2gray(exTestImage));

figure;

subplot(1,2,1);
imshow(exTestImage);


subplot(1,2,2);
imshow(processedImage);

% Using HOG features (Sýnýflandýrma yapabilmek için eðitim verilerinden HOG kullanýyoruz)

img = readimage(trainingSet, 11);

% Extract HOG features and HOG visualization
[hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',[2 2]);
[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
[hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);

% Show the original image
figure; 
subplot(2,3,1:3); imshow(img);

% Visualize the HOG features
subplot(2,3,4);  
plot(vis2x2); 
title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});

subplot(2,3,5);
plot(vis4x4); 
title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});

subplot(2,3,6);
plot(vis8x8); 
title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});

% iyi bir sonuç için 4*4 iþimize yarayacaktýr.
cellSize = [4 4];
hogFeatureSize = length(hog_4x4);


%HOG özelliklerini eðitim setinden çýkararak baþladýk ve TrainingSet'te
%dolaþýlýyo  her görüntüden HOG özellikleri çýkartýlýyor.

numImages2 = numel(trainingSet.Files);%matristeki toplam eleman sayýsý.(trainSet'teki resim sayýsý 10*10=100)
trainingFeatures = zeros(numImages2, hogFeatureSize, 'single');

for i = 1:numImages2
    img = readimage(trainingSet, i);
    
    img = rgb2gray(img);
    
    % Apply pre-processing steps
    img = imbinarize(img);
    
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end

% Get labels for each image.
trainingLabels = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures, trainingLabels);
[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(testSet, hogFeatureSize, cellSize);
predictedLabels = predict(classifier, testFeatures);
confMat = confusionmat(testLabels, predictedLabels);

helperDisplayConfusionMatrix(confMat)





%% (classification)  Hog gradient deðerleri ile uðraþtým fakat sýnýflandýrma yapamadým,uðraþtýðým kadarý 
%yorum satýrý olarak býrakýyorum.

%zero=0;one=0;two=0;three=0;four=0;five=0;six=0;seven=0;eight=0;nine=0;


%for i =1:10
 %   zero=zero+trainingFeatures(i,:);
  %  one=one+trainingFeatures(10+i,:);
   % two=two+trainingFeatures(20+i,:);
    %three=three+trainingFeatures(30+i,:);
    %four=four+trainingFeatures(40+i,:);
    %five=five+trainingFeatures(50+i,:);
    %six=six+trainingFeatures(60+i,:);
    %seven=seven+trainingFeatures(70+i,:);
    %eight=eight+trainingFeatures(80+i,:);
   % nine=nine+trainingFeatures(90+i,:);
%end

 
%mean
%label0=mean(zero/10);
%label1=mean(one/10);
%label2=mean(two/10);
%label3=mean(three/10);
%label4=mean(four/10);
%label5=mean(five/10);
%label6=mean(six/10);
%label7=mean(seven/10);
%label8=mean(eight/10);
%label9=mean(nine/10);

%% testing (test edeceðimiz resmin HOG feature'nu çýkarýyoruz)
%testimg = imread('test8.jpg');
%testimg2 = imbinarize(rgb2gray(testimg));
%[hog_test_4x4, vis_test_4x4] = extractHOGFeatures(testimg2,'CellSize',[4 4]);

%ortalama=mean(hog_test_4x4);

%%Karþýlaþtýrma

%classification=abs(hog_test_4x4 - three);



%%

