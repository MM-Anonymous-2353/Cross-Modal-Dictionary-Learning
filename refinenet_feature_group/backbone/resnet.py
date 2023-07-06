# """RefineNet-LightWeight

# RefineNet-LigthWeight PyTorch for non-commercial purposes

# Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# """

# import torch.nn as nn
# import torch.nn.functional as F
# import torch

# from .helpers import maybe_download
# from .layer_factory import conv1x1, conv3x3, CRPBlock

# data_info = {7: "Person", 21: "VOC", 40: "NYU", 60: "Context"}

# models_urls = {
#     "50_person": "https://cloudstor.aarnet.edu.au/plus/s/mLA7NxVSPjNL7Oo/download",
#     "101_person": "https://cloudstor.aarnet.edu.au/plus/s/f1tGGpwdCnYS3xu/download",
#     "152_person": "https://cloudstor.aarnet.edu.au/plus/s/Ql64rWqiTvWGAA0/download",
#     "50_voc": "https://cloudstor.aarnet.edu.au/plus/s/xp7GcVKC0GbxhTv/download",
#     "101_voc": "https://cloudstor.aarnet.edu.au/plus/s/CPRKWiaCIDRdOwF/download",
#     "152_voc": "https://cloudstor.aarnet.edu.au/plus/s/2w8bFOd45JtPqbD/download",
#     "50_nyu": "https://cloudstor.aarnet.edu.au/plus/s/gE8dnQmHr9svpfu/download",
#     "101_nyu": "https://cloudstor.aarnet.edu.au/plus/s/VnsaSUHNZkuIqeB/download",
#     "152_nyu": "https://cloudstor.aarnet.edu.au/plus/s/EkPQzB2KtrrDnKf/download",
#     "101_context": "https://cloudstor.aarnet.edu.au/plus/s/hqmplxWOBbOYYjN/download",
#     "152_context": "https://cloudstor.aarnet.edu.au/plus/s/O84NszlYlsu00fW/download",
#     "50_imagenet": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
#     "101_imagenet": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
#     "152_imagenet": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
# }

# stages_suffixes = {0: "_conv", 1: "_conv_relu_varout_dimred"}


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(
#             planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out


# class encoder(nn.Module):
#     def __init__(self, block, layers):
#         self.inplanes = 64
#         super(encoder, self).__init__()
#         self.do = nn.Dropout(p=0.5)
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.p_ims1d2_outl1_dimred = conv1x1(2048, 512, bias=False)
#         self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
#         self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)
#         self.p_ims1d2_outl2_dimred = conv1x1(1024, 256, bias=False)
#         self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
#         self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
#         self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

#         self.p_ims1d2_outl3_dimred = conv1x1(512, 256, bias=False)
#         self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
#         self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
#         self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

#         self.p_ims1d2_outl4_dimred = conv1x1(256, 256, bias=False)
#         self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
#         self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)

#         # self._init_weight()

#     def _make_crp(self, in_planes, out_planes, stages):
#         layers = [CRPBlock(in_planes, out_planes, stages)]
#         return nn.Sequential(*layers)

#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(
#                     self.inplanes,
#                     planes * block.expansion,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for i in range(1, blocks):
#             layers.append(block(self.inplanes, planes))

#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)

#         l1 = self.layer1(x)
#         l2 = self.layer2(l1)
#         l3 = self.layer3(l2)
#         l4 = self.layer4(l3)

#         l4 = self.do(l4)
#         l3 = self.do(l3)

#         x4 = self.p_ims1d2_outl1_dimred(l4)
#         x4 = self.relu(x4)
#         x4 = self.mflow_conv_g1_pool(x4)
#         x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
#         x4 = nn.Upsample(size=l3.size()[2:], mode="bilinear", align_corners=True)(x4)

#         x3 = self.p_ims1d2_outl2_dimred(l3)
#         x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
#         x3 = x3 + x4
#         x3 = F.relu(x3)
#         x3 = self.mflow_conv_g2_pool(x3)
#         x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
#         x3 = nn.Upsample(size=l2.size()[2:], mode="bilinear", align_corners=True)(x3)

#         x2 = self.p_ims1d2_outl3_dimred(l2)
#         x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
#         x2 = x2 + x3
#         x2 = F.relu(x2)
#         x2 = self.mflow_conv_g3_pool(x2)
#         x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
#         x2 = nn.Upsample(size=l1.size()[2:], mode="bilinear", align_corners=True)(x2)

#         x1 = self.p_ims1d2_outl4_dimred(l1)
#         x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
#         x1 = x1 + x2
#         x1 = F.relu(x1)
#         x1 = self.mflow_conv_g4_pool(x1)

#         return x1

#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

# class ResNetLW(nn.Module):
#     def __init__(self, block, layers, num_classes=21):
#         super(ResNetLW, self).__init__()
#         self.encoder_rgb = encoder(block, layers)
#         self.encoder_depth = encoder(block, layers)
#         # print(self.encoder_rgb)
#         # print(self.encoder_rgb.layer4[2].bn3)
#         # self.kernel_rgb = nn.Conv2d(256, 256, 3, padding=1, bias=False)
#         # self.kernel_depth = nn.Conv2d(256, 256, 3, padding=1, bias=False)
#         # self.bn_rgb = nn.GroupNorm(32, 256)
#         # self.bn_depth = nn.GroupNorm(32, 256)
#         self.kernel_rgb = Conv2d_Sparse(256, 256, 3, padding=1, num_sparsity_groups=4, num_sparsity_cardinals=16, bias=False)
#         self.kernel_depth = Conv2d_Sparse(256, 256, 3, padding=1, num_sparsity_groups=4, num_sparsity_cardinals=16, bias=False)
#         # self.downsample = nn.Sequential(
#         #     conv1x1(512, 256, bias=False),
#         #     nn.BatchNorm2d(256),
#         #     nn.ReLU(inplace=True))
#         # self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
#         self.clf_conv = nn.Conv2d(256 * 2, num_classes, kernel_size=3, stride=1, padding=1, bias=True)
#         # print(self.mflow_conv_g4_pool)
            
        

#         key = "152_imagenet"
#         url = models_urls[key]
#         self.encoder_rgb.load_state_dict(maybe_download(key, url), strict=False)
#         self.encoder_depth.load_state_dict(maybe_download(key, url), strict=False)
#         # self.downsample.apply(self._init_weight)
#         # self.clf_conv.apply(self._init_weight)

#     def _make_crp(self, in_planes, out_planes, stages):
#         layers = [CRPBlock(in_planes, out_planes, stages)]
#         return nn.Sequential(*layers)

#     def forward(self, rgbs, depths, masks):
#         rgb_input = torch.tensor([rgb.tolist() for rgb, mask1 in zip(rgbs, masks[:, 0]) if mask1 != 0]).to(rgbs.device)
#         depth_input = torch.tensor([depth.tolist() for depth, mask2 in zip(depths, masks[:, 1]) if mask2 != 0]).to(depths.device)
        
#         # input_shape = rgb_input.shape[-2:] if rgb_input.shape[0] != 0 else depth_input.shape[-2:]
        
#         input_shape = rgbs.shape[-2:]
        
#         rgb_not_miss_idx = torch.where(masks[:, 0] != 0)[0]
#         depth_not_miss_idx = torch.where(masks[:, 1] != 0)[0]
#         full_samples = torch.Tensor([idx for idx in rgb_not_miss_idx if idx in depth_not_miss_idx]).to(masks.device, dtype=torch.long)
#         full_sample_at_depth_loc = [depth_not_miss_idx.tolist().index(idx) for idx in full_samples]
#         full_sample_at_rgb_loc = [rgb_not_miss_idx.tolist().index(idx) for idx in full_samples]
#         feature_rgb = self.encoder_rgb(rgb_input) if rgb_input.shape[0] != 0 else None
#         feature_depth = self.encoder_depth(depth_input) if depth_input.shape[0] != 0 else None

#         # rgb_bn = self.bn_rgb(feature_rgb) if feature_rgb is not None else None
#         # depth_bn = self.bn_depth(feature_depth) if feature_depth is not None else None

#         # module_bn_rgb = self.bn_rgb
#         # module_bn_depth = self.bn_depth

#         # list_1 = [feature_rgb, module_bn_depth]
#         # list_2 = [feature_depth, module_bn_rgb]

#         if rgb_input.shape[0] != 0:
#             # feature_rgb = self.encoder_rgb(rgb_input)
#             pred_depth = self.kernel_depth(feature_rgb)
#         else:
#             # feature_rgb = None
#             pred_depth = None
            
#         if depth_input.shape[0] != 0:
#             # feature_depth = self.encoder_depth(depth_input)
#             pred_rgb = self.kernel_rgb(feature_depth)
#         else:
#             # feature_depth = None
#             pred_rgb = None
#         # feature_rgb = self.encoder_rgb(rgb_input) if rgb_input.shape[0] != 0 else None
#         # feature_depth = self.encoder_depth(depth_input) if depth_input.shape[0] != 0 else None
#         if feature_rgb is not None:
#             bs_input_shape = (masks.shape[0], feature_rgb.shape[1], feature_rgb.shape[2], feature_rgb.shape[3])
#         else:
#             bs_input_shape = (masks.shape[0], feature_depth.shape[1], feature_depth.shape[2], feature_depth.shape[3])

#         # if feature_rgb is not None:
#         #     pred_depth = self.kernel_depth(feature_rgb)
#         # else:
#         #     pred_depth = None
            
#         # if feature_depth is not None:
#         #     pred_rgb = self.kernel_rgb(feature_depth)
#         # else:
#         #     pred_rgb = None
        
#         if feature_rgb is not None:
#             expand_rgb = torch.zeros(bs_input_shape, requires_grad=True).to(feature_rgb.device)
#             expand_rgb[depth_not_miss_idx] = pred_rgb if feature_depth is not None else 0
#             expand_rgb[rgb_not_miss_idx] = feature_rgb
#         else:
#             expand_rgb = torch.zeros(bs_input_shape, requires_grad=True).to(feature_depth.device)
#             expand_rgb[depth_not_miss_idx] = pred_rgb
        
#         if feature_depth is not None:
#             expand_depth = torch.zeros(bs_input_shape, requires_grad=True).to(feature_depth.device)
#             expand_depth[rgb_not_miss_idx] = pred_depth if feature_rgb is not None else 0
#             expand_depth[depth_not_miss_idx] = feature_depth
#         else:
#             expand_depth = torch.zeros(bs_input_shape, requires_grad=True).to(feature_rgb.device)
#             expand_depth[rgb_not_miss_idx] = pred_depth
            
#         if full_samples.shape[0] != 0:
#             depth_pair = [pred_depth[full_sample_at_rgb_loc].to(feature_rgb.device), expand_depth[full_samples].to(feature_rgb.device)]
#             rgb_pair = [pred_rgb[full_sample_at_depth_loc].to(feature_depth.device), expand_rgb[full_samples].to(feature_depth.device)]
#         else:
#             depth_pair = [torch.zeros(bs_input_shape, requires_grad=True).to(expand_depth.device), torch.zeros(bs_input_shape, requires_grad=True).to(expand_depth.device)]
#             rgb_pair = [torch.zeros(bs_input_shape, requires_grad=True).to(expand_rgb.device), torch.zeros(bs_input_shape, requires_grad=True).to(expand_rgb.device)]
            
        

#         cat_feat = torch.cat([expand_rgb, expand_depth], dim=1)
        
    
#         # out = self.clf_conv(self.downsample(cat_feat))
#         out = self.clf_conv(cat_feat)
#         out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)

#         return out, rgb_pair, depth_pair


#     def _init_weight(self, m):
#         # for m in self.modules():
#         if isinstance(m, nn.Conv2d):
#             nn.init.kaiming_normal_(m.weight)
#         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
#             nn.init.constant_(m.weight, 1)
#             nn.init.constant_(m.bias, 0)


# def rf_lw50(num_classes, imagenet=False, pretrained=True, **kwargs):
#     model = ResNetLW(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
#     if imagenet:
#         key = "50_imagenet"
#         url = models_urls[key]
#         model.load_state_dict(maybe_download(key, url), strict=False)
#     elif pretrained:
#         dataset = data_info.get(num_classes, None)
#         if dataset:
#             bname = "50_" + dataset.lower()
#             key = "rf_lw" + bname
#             url = models_urls[bname]
#             model.load_state_dict(maybe_download(key, url), strict=False)
#     return model


# def rf_lw101(num_classes, imagenet=False, pretrained=True, **kwargs):
#     model = ResNetLW(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)
#     if imagenet:
#         key = "101_imagenet"
#         url = models_urls[key]
#         model.load_state_dict(maybe_download(key, url), strict=False)
#     elif pretrained:
#         dataset = data_info.get(num_classes, None)
#         if dataset:
#             bname = "101_" + dataset.lower()
#             key = "rf_lw" + bname
#             url = models_urls[bname]
#             model.load_state_dict(maybe_download(key, url), strict=False)
#     return model


# def rf_lw152(num_classes, imagenet=False, pretrained=True, **kwargs):
#     model = ResNetLW(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, **kwargs)
#     if imagenet:
#         key = "152_imagenet"
#         url = models_urls[key]
#         model.load_state_dict(maybe_download(key, url), strict=False)
#     elif pretrained:
#         dataset = data_info.get(num_classes, None)
#         if dataset:
#             bname = "152_" + dataset.lower()
#             key = "rf_lw" + bname
#             url = models_urls[bname]
#             model.load_state_dict(maybe_download(key, url), strict=False)
#     return model


# class Conv2d(nn.Conv2d):

#   def __init__(self,
#                in_channels,
#                out_channels,
#                kernel_size,
#                num_sparsity_groups=1,
#                num_sparsity_cardinals=1,
#                stride=1,
#                padding=0,
#                dilation=1,
#                groups=1,
#                bias=True,
#                padding_mode='zeros'):
#     self.num_sparsity_groups = num_sparsity_groups
#     self.num_sparsity_cardinals = num_sparsity_cardinals
#     out_channels *= num_sparsity_groups
#     super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
#                                  padding, dilation, groups, bias, padding_mode)
#     self.mlp = nn.Sequential(
#         nn.Linear(in_channels,
#                   self.num_sparsity_groups * self.num_sparsity_cardinals),)
#     if self.padding_mode == 'circular':
#       raise NotImplementedError

#   def _conv_forward(self, input, weight, bias):
#     sparsity = self.mlp(input.mean([-1, -2]))
#     sparsity = sparsity.view(
#         (-1, self.num_sparsity_groups, self.num_sparsity_cardinals))
#     sparsity = nn.functional.softmax(sparsity, dim=1)
#     weight = weight.view((self.num_sparsity_groups, self.num_sparsity_cardinals,
#                           -1, *weight.shape[1:]))
#     weight = torch.einsum("abc,bcdefg->acdefg", (sparsity, weight))
#     weight = weight.reshape((-1, *weight.shape[3:]))
#     if self.bias is not None:
#         bias = self.bias.view(self.num_sparsity_groups, self.num_sparsity_cardinals,
#                               -1)
#         bias = torch.einsum("abc,bcd->acd", (sparsity, bias))
#         bias = bias.reshape(-1)
#     batch_size = input.shape[0]
#     input = input.view((1, -1, *input.shape[2:]))
#     output = nn.functional.conv2d(input, weight, bias, self.stride,
#                                   self.padding, self.dilation,
#                                   self.groups * batch_size)
#     output = output.view((batch_size, -1, *output.shape[2:]))
#     return output

# class Conv2d_Sparse(nn.Conv2d):

#   def __init__(self,
#                in_channels,
#                out_channels,
#                kernel_size,
#                num_sparsity_groups=1,
#                num_sparsity_cardinals=1,
#                stride=1,
#                padding=0,
#                dilation=1,
#                groups=1,
#                bias=True,
#                padding_mode='zeros'):
#     self.num_sparsity_groups = num_sparsity_groups
#     self.num_sparsity_cardinals = num_sparsity_cardinals
#     super(Conv2d_Sparse, self).__init__(in_channels, out_channels, kernel_size, stride,
#                                  padding, dilation, groups, bias, padding_mode)

#     self.mlp = nn.Linear(in_channels, self.num_sparsity_groups * self.num_sparsity_cardinals)

#     if self.padding_mode == 'circular':
#       raise NotImplementedError

#   def _conv_forward(self, input, weight, bias):

#     coeff_input = input
#     # bn_w = input[1].weight.abs()
#     # bn_bias = input[1].bias

#     sparsity = self.mlp(coeff_input.mean([-1, -2]))
#     sparsity = sparsity.view((-1, self.num_sparsity_groups, self.num_sparsity_cardinals))
#     sparsity = nn.functional.softmax(sparsity, dim=1)
#     coeff_input = coeff_input.view((coeff_input.shape[0], self.num_sparsity_groups, self.num_sparsity_cardinals, -1, *coeff_input.shape[2:]))
#     sparse_input = torch.einsum("bkd,bkdehw->bkdehw", (sparsity, coeff_input))
#     sparse_input = sparse_input.reshape((coeff_input.shape[0], -1, *coeff_input.shape[4:]))

#     # weight = torch.einsum("i,ijhw->ijhw", (bn_w, weight))
#     # if bias is not None:
#     #     bias = bn_bias + bias

#     output = nn.functional.conv2d(sparse_input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
#     return output


"""RefineNet-LightWeight

RefineNet-LigthWeight PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

from .helpers import maybe_download
from .layer_factory import conv1x1, conv3x3, CRPBlock

data_info = {7: "Person", 21: "VOC", 40: "NYU", 60: "Context"}

models_urls = {
    "50_person": "https://cloudstor.aarnet.edu.au/plus/s/mLA7NxVSPjNL7Oo/download",
    "101_person": "https://cloudstor.aarnet.edu.au/plus/s/f1tGGpwdCnYS3xu/download",
    "152_person": "https://cloudstor.aarnet.edu.au/plus/s/Ql64rWqiTvWGAA0/download",
    "50_voc": "https://cloudstor.aarnet.edu.au/plus/s/xp7GcVKC0GbxhTv/download",
    "101_voc": "https://cloudstor.aarnet.edu.au/plus/s/CPRKWiaCIDRdOwF/download",
    "152_voc": "https://cloudstor.aarnet.edu.au/plus/s/2w8bFOd45JtPqbD/download",
    "50_nyu": "https://cloudstor.aarnet.edu.au/plus/s/gE8dnQmHr9svpfu/download",
    "101_nyu": "https://cloudstor.aarnet.edu.au/plus/s/VnsaSUHNZkuIqeB/download",
    "152_nyu": "https://cloudstor.aarnet.edu.au/plus/s/EkPQzB2KtrrDnKf/download",
    "101_context": "https://cloudstor.aarnet.edu.au/plus/s/hqmplxWOBbOYYjN/download",
    "152_context": "https://cloudstor.aarnet.edu.au/plus/s/O84NszlYlsu00fW/download",
    "50_imagenet": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "101_imagenet": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "152_imagenet": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

stages_suffixes = {0: "_conv", 1: "_conv_relu_varout_dimred"}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class encoder(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(encoder, self).__init__()
        self.do = nn.Dropout(p=0.5)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.p_ims1d2_outl1_dimred = conv1x1(2048, 512, bias=False)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv1x1(512, 256, bias=False)
        self.p_ims1d2_outl2_dimred = conv1x1(1024, 256, bias=False)
        self.adapt_stage2_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g2_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl3_dimred = conv1x1(512, 256, bias=False)
        self.adapt_stage3_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g3_pool = self._make_crp(256, 256, 4)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv1x1(256, 256, bias=False)

        self.p_ims1d2_outl4_dimred = conv1x1(256, 256, bias=False)
        self.adapt_stage4_b2_joint_varout_dimred = conv1x1(256, 256, bias=False)
        self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, stride=2, bias=False),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 1)
        )
        #self.conv = nn.Conv2d(256, 256, 1, bias=False)

        # self._init_weight()

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        l4 = self.do(l4)
        l3 = self.do(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = nn.Upsample(size=l3.size()[2:], mode="bilinear", align_corners=True)(x4)

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = x3 + x4
        x3 = F.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = nn.Upsample(size=l2.size()[2:], mode="bilinear", align_corners=True)(x3)

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = x2 + x3
        x2 = F.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = nn.Upsample(size=l1.size()[2:], mode="bilinear", align_corners=True)(x2)

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = x1 + x2
        x1 = F.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1_bn = self.conv_bn_relu(x1)
        x1_bn = nn.Upsample(size=x1.size()[2:], mode='bilinear', align_corners=True)(x1_bn)
        x0 = x1 + x1_bn
        #x1 = self.conv(x1)
        # x0 = F.relu(x0)
        x0 = self.relu(x0)

        return x0

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ResNetLW(nn.Module):
    def __init__(self, block, layers, num_classes=21, pre_ckpt=''):
        super(ResNetLW, self).__init__()
        self.encoder_rgb = encoder(block, layers)
        self.encoder_depth = encoder(block, layers)

        self.kernel_rgb1 = Conv2d_Sparse(256, 256, 3, reduction=16, padding=1, bias=False)
        self.kernel_depth1 = Conv2d_Sparse(256, 256, 3, reduction=16, padding=1, bias=False)
        # self.downsample = nn.Sequential(
        #     conv1x1(512, 256, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True))
        # self.mflow_conv_g4_pool = self._make_crp(256, 256, 4)
        self.clf_conv = nn.Conv2d(256 * 2, num_classes, kernel_size=3, stride=1, padding=1, bias=True)
            
        

        key = "152_imagenet"
        url = models_urls[key]
        self.encoder_rgb.load_state_dict(maybe_download(key, url), strict=False)
        self.encoder_depth.load_state_dict(maybe_download(key, url), strict=False)

        # checkpoint = torch.load(pre_ckpt)
        # print(pre_ckpt)
        # enc_state = {}
        # enc_rgb_state = {}
        # enc_depth_state = {}
        # for k, v in checkpoint["model_state"].items():
        #     # print(k)
        #     if 'encoder_rgb' in k:
        #         enc_rgb_state[k.split('encoder_rgb.')[1]] = v
        #     if 'encoder_depth' in k:
        #         enc_depth_state[k.split('encoder_depth.')[1]] = v
        #     if 'encoder_rgb' in k or 'encoder_depth' in k:
        #         enc_state[k] = v
        # self.encoder_rgb.load_state_dict(enc_rgb_state)
        # self.encoder_depth.load_state_dict(enc_depth_state)

    def _make_crp(self, in_planes, out_planes, stages):
        layers = [CRPBlock(in_planes, out_planes, stages)]
        return nn.Sequential(*layers)

    def forward(self, rgbs, depths, masks):
        rgb_input = torch.tensor([rgb.tolist() for rgb, mask1 in zip(rgbs, masks[:, 0]) if mask1 != 0]).to(rgbs.device)
        depth_input = torch.tensor([depth.tolist() for depth, mask2 in zip(depths, masks[:, 1]) if mask2 != 0]).to(depths.device)
        
        # input_shape = rgb_input.shape[-2:] if rgb_input.shape[0] != 0 else depth_input.shape[-2:]
        
        input_shape = rgbs.shape[-2:]
        
        rgb_not_miss_idx = torch.where(masks[:, 0] != 0)[0]
        depth_not_miss_idx = torch.where(masks[:, 1] != 0)[0]
        full_samples = torch.Tensor([idx for idx in rgb_not_miss_idx if idx in depth_not_miss_idx]).to(masks.device, dtype=torch.long)
        full_sample_at_depth_loc = [depth_not_miss_idx.tolist().index(idx) for idx in full_samples]
        full_sample_at_rgb_loc = [rgb_not_miss_idx.tolist().index(idx) for idx in full_samples]
        feature_rgb_orig = self.encoder_rgb(rgb_input) if rgb_input.shape[0] != 0 else None
        feature_depth_orig = self.encoder_depth(depth_input) if depth_input.shape[0] != 0 else None

        # rgb_bn = self.bn_rgb(feature_rgb) if feature_rgb is not None else None
        # depth_bn = self.bn_depth(feature_depth) if feature_depth is not None else None
        feat_shape = feature_rgb_orig.shape[-2:] if rgb_input.shape[0] != 0 else feature_depth_orig.shape[-2:]
        
        feature_rgb = F.interpolate(feature_rgb_orig, size=(feat_shape[0]//2, feat_shape[1]//2), mode='bilinear', align_corners=False) if rgb_input.shape[0] != 0 else None
        feature_depth = F.interpolate(feature_depth_orig, size=(feat_shape[0]//2, feat_shape[1]//2), mode='bilinear', align_corners=False) if depth_input.shape[0] != 0 else None

        module_bn_rgb = self.encoder_rgb.conv_bn_relu[1]
        module_bn_depth = self.encoder_depth.conv_bn_relu[1]
        # print(module_bn_rgb.weight)

        list_1 = [feature_rgb, module_bn_depth]
        list_2 = [feature_depth, module_bn_rgb]

        if rgb_input.shape[0] != 0:
            # feature_rgb = self.encoder_rgb(rgb_input)
            pred_depth, sparse_depth, sim_depth = self.kernel_depth1(list_1)
            pred_depth = F.interpolate(pred_depth, size=feat_shape, mode='bilinear', align_corners=False)
        else:
            # feature_rgb = None
            pred_depth = None
            sparse_depth = torch.zeros((depth_input.shape[0], 256), requires_grad=True).to(depth_input.device)
            
        if depth_input.shape[0] != 0:
            # feature_depth = self.encoder_depth(depth_input)
            pred_rgb, sparse_rgb, sim_rgb = self.kernel_rgb1(list_2)
            pred_rgb = F.interpolate(pred_rgb, size=feat_shape, mode='bilinear', align_corners=False)
        else:
            # feature_depth = None
            pred_rgb = None
            sparse_rgb = torch.zeros((rgb_input.shape[0], 256), requires_grad=True).to(rgb_input.device)
            
        # feature_rgb = self.encoder_rgb(rgb_input) if rgb_input.shape[0] != 0 else None
        # feature_depth = self.encoder_depth(depth_input) if depth_input.shape[0] != 0 else None
        if feature_rgb is not None:
            bs_input_shape = (masks.shape[0], feature_rgb_orig.shape[1], feature_rgb_orig.shape[2], feature_rgb_orig.shape[3])
        else:
            bs_input_shape = (masks.shape[0], feature_depth_orig.shape[1], feature_depth_orig.shape[2], feature_depth_orig.shape[3])

        # if feature_rgb is not None:
        #     pred_depth = self.kernel_depth(feature_rgb)
        # else:
        #     pred_depth = None
            
        # if feature_depth is not None:
        #     pred_rgb = self.kernel_rgb(feature_depth)
        # else:
        #     pred_rgb = None
        
        
        if feature_rgb is not None:
            expand_rgb = torch.zeros(bs_input_shape, requires_grad=True).to(feature_rgb.device)
            expand_rgb[depth_not_miss_idx] = pred_rgb if feature_depth is not None else 0
            expand_rgb[rgb_not_miss_idx] = feature_rgb_orig
        else:
            expand_rgb = torch.zeros(bs_input_shape, requires_grad=True).to(feature_depth.device)
            expand_rgb[depth_not_miss_idx] = pred_rgb
        
        if feature_depth is not None:
            expand_depth = torch.zeros(bs_input_shape, requires_grad=True).to(feature_depth.device)
            expand_depth[rgb_not_miss_idx] = pred_depth if feature_rgb is not None else 0
            expand_depth[depth_not_miss_idx] = feature_depth_orig
        else:
            expand_depth = torch.zeros(bs_input_shape, requires_grad=True).to(feature_rgb.device)
            expand_depth[rgb_not_miss_idx] = pred_depth
            
        if full_samples.shape[0] != 0:
            depth_pair = [pred_depth[full_sample_at_rgb_loc].to(feature_rgb.device), expand_depth[full_samples].to(feature_rgb.device)]
            rgb_pair = [pred_rgb[full_sample_at_depth_loc].to(feature_depth.device), expand_rgb[full_samples].to(feature_depth.device)]
        else:
            depth_pair = [torch.zeros(bs_input_shape, requires_grad=True).to(expand_depth.device), torch.zeros(bs_input_shape, requires_grad=True).to(expand_depth.device)]
            rgb_pair = [torch.zeros(bs_input_shape, requires_grad=True).to(expand_rgb.device), torch.zeros(bs_input_shape, requires_grad=True).to(expand_rgb.device)]
            
        sparse_list = [sparse_rgb, sparse_depth]
        if full_samples.shape[0] != 0:
            sim_list = [sim_rgb[full_sample_at_depth_loc], sim_depth[full_sample_at_rgb_loc]]
        else:
            sim_list = [torch.zeros(size=(bs_input_shape[0], bs_input_shape[1], feat_shape[0]//2, feat_shape[1]//2), requires_grad=True).to(expand_rgb.device), 
                        torch.zeros(size=(bs_input_shape[0], bs_input_shape[1], feat_shape[0]//2, feat_shape[1]//2) , requires_grad=True).to(expand_rgb.device)]

        cat_feat = torch.cat([expand_rgb, expand_depth], dim=1)
        
    
        # out = self.clf_conv(self.downsample(cat_feat))
        out = self.clf_conv(cat_feat)
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=False)

        return out, rgb_pair, depth_pair, sparse_list, sim_list


    def _init_weight(self, m):
        # for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
                # nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.weight, 1.0)


def rf_lw50(num_classes, imagenet=False, pretrained=True, **kwargs):
    model = ResNetLW(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)
    if imagenet:
        key = "50_imagenet"
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = "50_" + dataset.lower()
            key = "rf_lw" + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    return model


def rf_lw101(num_classes, imagenet=False, pretrained=True, **kwargs):
    model = ResNetLW(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)
    if imagenet:
        key = "101_imagenet"
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = "101_" + dataset.lower()
            key = "rf_lw" + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    return model


def rf_lw152(num_classes, imagenet=False, pretrained=True, pre_ckpt='', **kwargs):
    model = ResNetLW(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, pre_ckpt=pre_ckpt, **kwargs)
    if imagenet:
        key = "152_imagenet"
        url = models_urls[key]
        model.load_state_dict(maybe_download(key, url), strict=False)
    elif pretrained:
        dataset = data_info.get(num_classes, None)
        if dataset:
            bname = "152_" + dataset.lower()
            key = "rf_lw" + bname
            url = models_urls[bname]
            model.load_state_dict(maybe_download(key, url), strict=False)
    return model


class Conv2d(nn.Conv2d):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               num_sparsity_groups=1,
               num_sparsity_cardinals=1,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               bias=True,
               padding_mode='zeros'):
    self.num_sparsity_groups = num_sparsity_groups
    self.num_sparsity_cardinals = num_sparsity_cardinals
    out_channels *= num_sparsity_groups
    super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                 padding, dilation, groups, bias, padding_mode)
    self.mlp = nn.Sequential(
        nn.Linear(in_channels,
                  self.num_sparsity_groups * self.num_sparsity_cardinals),)
    if self.padding_mode == 'circular':
      raise NotImplementedError

  def _conv_forward(self, input, weight, bias):
    sparsity = self.mlp(input.mean([-1, -2]))
    sparsity = sparsity.view(
        (-1, self.num_sparsity_groups, self.num_sparsity_cardinals))
    sparsity = nn.functional.softmax(sparsity, dim=1)
    weight = weight.view((self.num_sparsity_groups, self.num_sparsity_cardinals,
                          -1, *weight.shape[1:]))
    weight = torch.einsum("abc,bcdefg->acdefg", (sparsity, weight))
    weight = weight.reshape((-1, *weight.shape[3:]))
    if self.bias is not None:
        bias = self.bias.view(self.num_sparsity_groups, self.num_sparsity_cardinals,
                              -1)
        bias = torch.einsum("abc,bcd->acd", (sparsity, bias))
        bias = bias.reshape(-1)
    batch_size = input.shape[0]
    input = input.view((1, -1, *input.shape[2:]))
    output = nn.functional.conv2d(input, weight, bias, self.stride,
                                  self.padding, self.dilation,
                                  self.groups * batch_size)
    output = output.view((batch_size, -1, *output.shape[2:]))
    return output

class Conv2d_Sparse(nn.Conv2d):

  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               reduction=16,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               bias=True,
               padding_mode='zeros'):
    super(Conv2d_Sparse, self).__init__(in_channels, out_channels, kernel_size, stride,
                                 padding, dilation, groups, bias, padding_mode)

    # self.mlp = nn.Linear(in_channels, self.num_sparsity_groups * self.num_sparsity_cardinals)
    # self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.mlp = nn.Sequential(
        nn.Linear(in_channels, in_channels//reduction, bias=False),
        # nn.ReLU(inplace=True),
        nn.Linear(in_channels//reduction, out_channels, bias=False),
        # nn.Sigmoid()
    )
    self.mlp_bn = nn.Linear(in_channels, out_channels)

    if self.padding_mode == 'circular':
      raise NotImplementedError

  def _conv_forward(self, input, weight, bias):

    coeff_input = input[0]
    # b, c = coeff_input.shape[:2]
    # sparsity = self.mlp(self.avg_pool(coeff_input).view(b,c))
    sparsity = self.mlp(coeff_input.mean([-1, -2]))
    sparsity = sparsity.abs()


    bn_w = input[1].weight.abs()
    bn_w = self.mlp_bn(bn_w)
    bn_w = nn.functional.sigmoid(bn_w)
   

    sparse_input = torch.einsum("bk,bkhw->bkhw", (sparsity, coeff_input))

    weight = torch.einsum("i,ijhw->ijhw", (bn_w, weight))
    
    # if bias is not None:
        # bias = bn_bias + bias

    output = nn.functional.conv2d(sparse_input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
    return output, sparsity, sparse_input