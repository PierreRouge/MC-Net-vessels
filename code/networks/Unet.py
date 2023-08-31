#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 15:56:46 2022

@author: rouge
Re-implementation of U-Net
"""


import torch
from torch import nn

#%% Layers

class Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, strides, kernel_size):
        
        super().__init__()
        
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, stride=strides, kernel_size=kernel_size, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        
    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        
        return x
    
class Convolution_trans(nn.Module):
    def __init__(self, in_channels, out_channels, strides, kernel_size):
        
        super().__init__()
        
        self.conv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, stride=strides, kernel_size=kernel_size, padding=1, output_padding=1)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        
        return x
        
        
    

class DoubleConv(nn.Sequential):
    def __init__(self, dim, in_features, out_features, strides, kernel_size):
        
        super().__init__()
        
        conv1 = Convolution(
            in_channels=in_features,
            out_channels=out_features,
            strides=strides,
            kernel_size=kernel_size,
            
        )
        
        conv2 = Convolution(
            in_channels=out_features,
            out_channels=out_features,
            strides=1,
            kernel_size=kernel_size,
        )
        
        self.add_module('conv1', conv1)
        self.add_module('conv2', conv2)
        
        
class Conv_Up_with_skip(nn.Module):
    def __init__(self, dim, in_features, out_features, strides, kernel_size):
        
        super().__init__()
        
        self.conv_trans = Convolution_trans(
            in_channels=in_features,
            out_channels=out_features,
            strides=strides,
            kernel_size=kernel_size,
            
        )
        
        self.conv1 = Convolution(
            in_channels=out_features * 2,
            out_channels=out_features,
            strides=(1, 1, 1),
            kernel_size=kernel_size,
            
        )
        
        self.conv2 = Convolution(
            in_channels=out_features,
            out_channels=out_features,
            strides=(1, 1, 1),
            kernel_size=kernel_size,
        )
        
    def forward(self, x, x_encoder):
        x_0 = self.conv_trans(x)
        x_1 = self.conv1(torch.cat([x_encoder, x_0], dim=1))
        x_2 = self.conv2(x_1)

        return x_2
    

#%% Modules


class ShallowEncoder(nn.Module):
    def __init__(self, dim, in_channel, features, strides, kernel_size):
        super(ShallowEncoder, self).__init__()

        self.conv1 = DoubleConv(dim, in_channel, features[0], strides[0], kernel_size[0])
        self.conv2 = DoubleConv(dim, features[0], features[1], strides[1], kernel_size[1])
        self.conv3 = DoubleConv(dim, features[1], features[2], strides[2], kernel_size[2])
        self.conv4 = DoubleConv(dim, features[2], features[3], strides[3], kernel_size[3])

    def forward(self, x: torch.Tensor):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        return x4, x3, x2, x1


class DeepEncoder(nn.Module):
    def __init__(self, dim, in_channel, features, strides, kernel_size):
        super(DeepEncoder, self).__init__()

        self.conv5 = DoubleConv(dim, features[3], features[4], strides[4], kernel_size[4])
        self.conv6 = DoubleConv(dim, features[4], features[5], strides[5], kernel_size[5])

    def forward(self, x4: torch.Tensor):

        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        return x6, x5



class TinyShallowEncoder(nn.Module):
    def __init__(self, dim, in_channel, features, strides, kernel_size):
        super(TinyShallowEncoder, self).__init__()

        self.conv1 = DoubleConv(dim, in_channel, features[0], strides[0], kernel_size[0])
        self.conv2 = DoubleConv(dim, features[0], features[1], strides[1], kernel_size[1])


    def forward(self, x: torch.Tensor):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        return x2, x1


class TinyDeepEncoder(nn.Module):
    def __init__(self, dim, in_channel, features, strides, kernel_size):
        super(TinyDeepEncoder, self).__init__()

        self.conv5 = DoubleConv(dim, features[1], features[2], strides[2], kernel_size[2])
        self.conv6 = DoubleConv(dim, features[2], features[3], strides[3], kernel_size[3])

    def forward(self, x2: torch.Tensor):

        x3 = self.conv5(x2)
        x4 = self.conv6(x3)

        return x4, x3

    
class TinyEncoder(nn.Module):
    def __init__(self, dim, in_channel, features, strides, kernel_size):
        super(TinyEncoder, self).__init__()

        self.shallowencoder = TinyShallowEncoder(dim, in_channel, features, strides, kernel_size)
        self.deepencoder = TinyDeepEncoder(dim, in_channel, features, strides, kernel_size)
        

    def forward(self, x: torch.Tensor):

        x1, _ = self.shallowencoder(x)
        x2, _ = self.deepencoder(x1)

        return x2
    
    def freeze(self):
        for p in self.shallowencoder.parameters():
            p.requires_grad = False
        for p in self.deepencoder.parameters():
            p.requires_grad = False
    
    def load_shallowencoder_weights(self, shallowencoder_weights):
        self.shallowencoder.load_state_dict(shallowencoder_weights)

    def load_deepencoder_weights(self, deepencoder_weights):
        self.deepencoder.load_state_dict(deepencoder_weights)
        
    def load_decoder_weights(self, shallowencoder_weights, deepencoder_weights):
        self.shallowencoder.load_state_dict(shallowencoder_weights)
        self.deepencoder.load_state_dict(deepencoder_weights)
        
    def get_shallowencoder_weights(self):
        return self.shallowencoder.state_dict()
    
    def get_deepencoder_weights(self):
        return self.deepencoder.state_dict()
        
    def dim_latent_space(self, x: torch.Tensor):
        
        with torch.no_grad():
            x1, _ = self.shallowencoder(x)
            x2, _ = self.deepencoder(x1)

        return x2.shape



class miniShallowEncoder(nn.Module):
    def __init__(self, dim, in_channel, features, strides, kernel_size):
        super(miniShallowEncoder, self).__init__()

        self.conv1 = DoubleConv(dim, in_channel, features[0], strides[0], kernel_size[0])
        self.conv2 = DoubleConv(dim, features[0], features[1], strides[1], kernel_size[1])


    def forward(self, x: torch.Tensor):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        return x2, x1



class Decoder_UNet(nn.Module):
    def __init__(self, dim, in_channel, features, strides, kernel_size):
        super(Decoder_UNet, self).__init__()
        
        self.up_1 = Conv_Up_with_skip(dim, features[5], features[4], strides[5], kernel_size[5])
        self.up_2 = Conv_Up_with_skip(dim, features[4], features[3], strides[4], kernel_size[4])
        self.up_3 = Conv_Up_with_skip(dim, features[3], features[2], strides[3], kernel_size[3])
        self.up_4 = Conv_Up_with_skip(dim, features[2], features[1], strides[2], kernel_size[2])
        self.up_5 = Conv_Up_with_skip(dim, features[1], features[0], strides[1], kernel_size[1])
        
        self.final_conv_1 = nn.Conv3d(features[0], 1, kernel_size=1)
        
    def forward(self, x6: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor):
            
        x_7 = self.up_1(x6, x5)
        x_8 = self.up_2(x_7, x4)
        x_9 = self.up_3(x_8, x3)
        x_10 = self.up_4(x_9, x2)
        x_11 = self.up_5(x_10, x1)
        x_final = self.final_conv_1(x_11)
            
        return x_final


class TinyDecoder_UNet(nn.Module):
    
    def __init__(self, dim, in_channel, features, strides, kernel_size, nclasses):
        super(TinyDecoder_UNet, self).__init__()
        self.up_1 = Conv_Up_with_skip(dim, features[3], features[2], strides[3], kernel_size[3])
        self.up_2 = Conv_Up_with_skip(dim, features[2], features[1], strides[2], kernel_size[2])
        self.up_3 = Conv_Up_with_skip(dim, features[1], features[0], strides[1], kernel_size[1])
        self.final_conv_1 = nn.Conv3d(features[0], nclasses, kernel_size=1)

    def forward(self, x4: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor):
        x5 = self.up_1(x4, x3)
        x6 = self.up_2(x5, x2)
        x7 = self.up_3(x6, x1)
        x_final = self.final_conv_1(x7)

        return x_final

class TinyDiscriminator(nn.Module):
    def __init__(self, in_features, nbr_classes):
        super(TinyDiscriminator, self).__init__()

        self.conv1 = nn.Conv3d(256, 128, 3, padding='same')
        self.conv2 = nn.Conv3d(128, 64, 3, padding='same')
        self.conv3 = nn.Conv3d(64, 32, 3, padding='same')
        self.conv4 = nn.Conv3d(32, 16, 3, padding='same')
        self.conv5 = nn.Conv3d(16, 8, 3, padding='same')
        self.conv6 = nn.Conv3d(8, 4, 3, padding='same')

        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(in_features, in_features // 2)
        self.layer2 = nn.Linear(in_features // 2, in_features // 4)
        self.layer3 = nn.Linear(in_features // 4, 500)
        self.final = nn.Linear(500, 1)

    def forward(self, x4: torch.Tensor):
        
        x5 = self.conv1(x4)
        x6 = self.conv2(x5)
        x7 = self.conv3(x6)
        x8 = self.conv4(x7)
        x9 = self.conv5(x8)
        x10 = self.conv6(x9)
        
        x11 = self.flatten(x10)
        x12 = self.layer1(x11)
        x13 = self.layer2(x12)
        x14 = self.layer3(x13)
        x_final = self.final(x14)

        return x_final
#%% Models

      
class TinyUnet(nn.Module):
    def __init__(self, dim, in_channel, features, strides, kernel_size, nclasses):
        super(TinyUnet, self).__init__()

        self.shallowencoder = TinyShallowEncoder(dim, in_channel, features, strides, kernel_size)
        self.deepencoder = TinyDeepEncoder(dim, in_channel, features, strides, kernel_size)
        
        self.decoder1 = TinyDecoder_UNet(dim, in_channel, features, strides, kernel_size, nclasses)
        self.decoder2 = TinyDecoder_UNet(dim, in_channel, features, strides, kernel_size, nclasses)


    def forward(self, x: torch.Tensor):
        
        x2, x1 = self.shallowencoder(x)
        x4, x3 = self.deepencoder(x2)
        
        x_final_1  = self.decoder1(x4, x1, x2, x3)
        x_final_2  = self.decoder2(x4, x1, x2, x3)
        return x_final_1, x_final_2

    def load_shallowencoder_weights(self, shallowencoder_weights):
        self.shallowencoder.load_state_dict(shallowencoder_weights)

    def load_deepencoder_weights(self, deepencoder_weights):
        self.deepencoder.load_state_dict(deepencoder_weights)
        
    def get_shallowencoder_weights(self):
        return self.shallowencoder.state_dict()

    def get_deepencoder_weights(self):
        return self.deepencoder.state_dict()

    def freeze_shallowencoder_weights(self):
        for p in self.shallowencoder.parameters():
            p.requires_grad = False

    def freeze_deepencoder_weights(self):
        for p in self.deepencoder.parameters():
            p.requires_grad = False


class BaseUnet(nn.Module):
     
    def __init__(self, dim, in_channel, features, strides, kernel_size):
        super(BaseUnet, self).__init__()
        
        self.shallowencoder = ShallowEncoder(dim, in_channel, features, strides, kernel_size)
        self.deepencoder = DeepEncoder(dim, in_channel, features, strides, kernel_size)
        self.decoder = Decoder_UNet(dim, in_channel, features, strides, kernel_size)
        
    def forward(self, x: torch.Tensor):
            
        x4, x3, x2, x1 = self.shallowencoder(x)
        x6, x5 = self.deepencoder(x4)
        x_final = self.decoder(x6, x1, x2, x3, x4, x5)
            
        return x_final
    
    def load_shallowencoder_weights(self, shallowencoder_weights):
        self.shallowencoder.load_state_dict(shallowencoder_weights)
        
    def load_deepencoder_weights(self, deepencoder_weights):
        self.deepencoder.load_state_dict(deepencoder_weights)
    
    def freeze_shallowencoder_weights(self):
        for p in self.shallowencoder.parameters():
            p.requires_grad = False
            
    def freeze_deepencoder_weights(self):
        for p in self.deepencoder.parameters():
            p.requires_grad = False
       
            

  
  
