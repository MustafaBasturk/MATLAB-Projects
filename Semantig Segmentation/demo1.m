%Bu projenin içeriği, semantic segmentation(anlamlı bölütleme) kullanarak bir
%resimdeki nesneyi, nesnenin her pikseline kadar etiketlemeyi gösterir.

%Kullandığım bu örnek ise daha önceden eğitilmiş olan Deeplab v3+ ağını
%kullanarak bir görüntüyü nasıl bölümlere ayıracağını gösterir.

%Eğitim prosedürünü göstermek için  Cambridge Üniversitesinden
%daha önceden eğitilmiş olan bir veri setini kullandım.

%İlk olarak orijinal resim verilerini ve etiketlenmiş resim verilerini
%yüklüyorum.
imgDir = fullfile('CamVid', 'images');
lblDir = fullfile('CamVid', 'labels');

%Önceden eğitilmiş olan network'ü yüklüyorum.
pretrainedNetwork = fullfile('deeplabv3plusResnet18CamVid','deeplabv3plusResnet18CamVid.mat');  
data = load(pretrainedNetwork);
net = data.net;

%Bölütlemede kullanılacak sınıflarımı belirliyorum. 
classes = [    
    "Sky"
    "Building"
    "Pole"
    "Road"
    "Pavement"
    "Tree"
    "SignSymbol"
    "Fence"
    "Car"
    "Pedestrian"
    "Bicyclist"
    ];

%Bölütleme için renk haritasını ve pixellenmiş etiketli resimlerin idsini 
% belirliyorum.
cmap = camvidColorMap();
pixelLabelID = camvidPixelLabelIDs();

%Datastore kodumla, orijinal ve etiketlenmiş resimleri büyük veri kümeleri
%olmasından dolayı işlemek için bir depolama nesnesi oluşturuyorum.
%Kısacası büyük veri dosyalarını işlememde kolaylık sağlıyor.
imgds = imageDatastore(imgDir);
pxlds = pixelLabelDatastore(lblDir,classes,pixelLabelID);

I = readimage(imgds,231);
I = histeq(I);

C = readimage(pxlds,231);

B = labeloverlay(I,C,'ColorMap',cmap);
figure(1)
imshow(B)
pixelLabelColorbar(cmap,classes);

%Kod ne kadar önceden eğitilmiş olsa da etiketleme ve renklendirmede
%iyileştirmeler gerektirir. 


%Kodun eğitimini ve doğruluğunu arttırmak için DeepLab v3+'ın
%fonksiyonlarından partitionCamVidData'yı kullanıyoruz. Bu kod resimlerin
% %60'ını eğitimde kullanırken geri kalanını da test için kullanır.
[imgdsTrain, imgdsTest, pxldsTrain, pxldsTest] = partitionCamVidData(imgds, pxlds);

%60 eğitimde kullanılan:
numTrainingImages = numel(imgdsTrain.Files)

%40 testte kullanılan:
numTestImages = numel(imgdsTest.Files)

%Network'u kurma aşamasına giriyoruz.

%Eğitimdeki kullanılan resimlerle aynı boyutu kullanmak için:
imageSize = [720 960 3];

%Sınıfların numarasını belirlemek için:
numClasses = numel(classes);

% DeepLab v3+ network oluşturma:
lgraph = deeplabv3plusLayers(imageSize, numClasses, "resnet18");

% analyzeNetwork(lgraph)

%Sınıfların dengesiz olmasından dolayı sınıf ağırlıklandrması kullanmak
%gerekir. 

%dsTrain, imgdsTrain ve pxldsTrain veri kümesini birleştirerek eğitim veri
%kümesini oluşturuyor.
dsTrain = combine(imgdsTrain, pxldsTrain);
%tbl, pxlds içerisindeki etiketlenmiş veriler için sınıf frekanslarını
%hesaplar.
tbl = countEachLabel(pxlds);
%imageFreq, her sınıfın veri kümesindeki piksel sayısını tutar. 
imageFreq = tbl.ImagePixelCount;
%classWeight, Sınıf ağırlıklarını belirlemek için frekans değerlerini alır.
classWeight = imageFreq
%pxLayer, piksel sınıflandırma katmanını oluşturur.
pxLayer = pixelClassificationLayer('Name','Labels','Classes',tbl.Name,'ClassWeights',classWeight);
%lgraph, lgraph ile pxLayer ile yer değiştirir. Bunun sebebi modelin çıkışı
%piksel bazında sınıflandırma yapacak şekilde güncellenmesidir.
lgraph = replaceLayer(lgraph, "classification", pxLayer);

%Eğitim için kullanılan optimizasyon algoritması, momentumlu stokastik gradyan inişidir (SGDM).
%https://towardsdatascience.com/stochastic-gradient-descent-with-momentum-a84097641a5d
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...    
    'MaxEpochs',30, ...  
    'MiniBatchSize',8, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 4);

%Bu aşamadan sonrası veri artırma işlemidir. Veri artırma işlemi, etiketli
%eğitim verilerinin sayısını artırmadan çeşitliliği artırmak için
%kullanılır.
dsTrain = combine(imgdsTrain, pxldsTrain);

%Burada, veri büyütme için rastgele sol/sağ yansıma ve +/- 10 pikselin
% rastgele X/Y çevirisi kullanılır. Sonrasında veri artırma işlemi
% gerçekleştirilir.
xTrans = [-10 10];
yTrans = [-10 10];
dsTrain = transform(dsTrain, @(data)augmentImageAndLabel(data, xTrans, yTrans));
%Bu işlemlerin test ve doğrulama verileri üzerinden yapıldığı da
%unutulmamalıdır.

%Eğitime başlamak için trainNetwork komutu kullanıyoruz.
%Not: Eğitim işlemi  NVIDIA™ Titan X with 12 GB of GPU memory cihazıyla
%test edildiğinden daha düşük teknolojili cihazlarda işlem uzun
%sürebiliyor.
doTraining = false;
if doTraining
    [net, info] = trainNetwork(dsTrain,lgraph,options);
end

%Ağımızı eğitilmiş veriler üzerinde test ediyoruz.
I = readimage(imgdsTest,1);
C = semanticseg(I, net);

B = labeloverlay(I,C,'Colormap',cmap,'Transparency',0.4);
figure(2)
imshow(B)
pixelLabelColorbar(cmap, classes);

%Sonucu karşılaştırıyoruz.
% expectedResult = readimage(pxldsTest,1);
% actual = uint8(C);
% expected = uint8(expectedResult);
% figure(3)
% imshowpair(actual, expected);
