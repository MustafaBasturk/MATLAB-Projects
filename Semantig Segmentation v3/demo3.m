%Önceden eğitilmiş model yüklenir.
pretrainedNetwork = fullfile('deeplabv3plusResnet18CamVid','deeplabv3plusResnet18CamVid.mat');    
data = load(pretrainedNetwork);
net = data.net;

%Bölütleme uygulanacak resim seçilir.
imgDir = fullfile('images', '1.png');
img = imread(imgDir);

%Resim, modelin beklediği giriş boyutuna yeniden boyutlandırılır.
inputSize = net.Layers(1).InputSize;
desiredSize = inputSize(1:2);
img = imresize(img, desiredSize);

%Görüntü üzerinde segmentasyon uygulanır.
segmentationMap = semanticseg(img, net);

%Orijinal görüntü üzerine segmentasyon haritası bindirilir.
B = labeloverlay(img, segmentationMap);
figure(1)
imshow(B);

figure(2)
imshow(img);