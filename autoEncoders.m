%% This script uses a stacked autoencoder in first reducing the
%  dimensions of the data

% cd 'Lec 6 - Model Order Reduction 1\matlab code and brain image mat file'

% Load the blurred Shepp-Logan Images
load('Datasets/imageVectors.mat','G','Gb','GTest','x','xTest','nx','ny','mSize','g')

imshow(imresize(Gb(:,:,1),30))

% Train the first autoencoder with hiddenSize = 10
hiddenSize1 = 70;
autoenc1 = trainAutoencoder(x, hiddenSize1);

% Show the encoding and reconstructed data
z1 = encode(autoenc1, x);
xR1 = decode(autoenc1, z1);
% montage([imresize(Gb(:,:,6),30), imresize(reshape(xR1(:,6),nx,ny),30)])
% imshow(imresize(reshape(xR(:,1),nx,ny),30));

% Pass the encoded data to a 10-D encoder
hiddenSize2 = 30;
autoenc2 = trainAutoencoder(z1, hiddenSize2);

% Check the encoding and reconstructed data again
z2 = encode(autoenc2, z1);
xR2 = decode(autoenc2, z2);
figure;
% montage([imresize(Gb(:,:,6),30), imresize(reshape(xR1(:,6),nx,ny),30),...
%                                 imresize(reshape(xR2(:,6),nx,ny),30)])

% xR3 = decode(autoenc1, xR2);
% montage([imresize(Gb(:,:,6),30), imresize(reshape(xR1(:,6),nx,ny),30),...
%                                 imresize(reshape(xR3(:,6),nx,ny),30)])
                            
% Create a stacked network to automate the last few lines
stackednet = stack(autoenc1, autoenc2);

% Test on the test image
xTestR = stackednet(xTest);
xTestR1 = decode(autoenc1, decode(autoenc2, xTestR));
xTestR2 = decode(autoenc1, decode(autoenc2, encode(autoenc2, encode(autoenc1, xTest))));
montage([reshape(xTest(:,1),nx,ny), reshape(xTestR1(:,1),nx,ny), reshape(xTestR2(:,1),nx,ny)]);