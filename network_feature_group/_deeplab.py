import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from .utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass

class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project_rgb = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.project_depth = nn.Sequential( 
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.low_transition_rgb = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            # nn.LayerNorm((256, 24, 48)),
            nn.ReLU(inplace=True)
        )
        self.low_transition_depth = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            # nn.LayerNorm((256, 24, 48)),
            nn.ReLU(inplace=True)
        )


        self.aspp_rgb = ASPP(in_channels, aspp_dilate)
        self.aspp_depth = ASPP(in_channels, aspp_dilate)
        
        self.transition_rgb = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            # nn.LayerNorm((256, 24, 48)),
            nn.ReLU(inplace=True)
        )
        self.transition_depth = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            # nn.LayerNorm((256, 24, 48)),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(256 * 2, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fianl = nn.Conv2d(352, num_classes, 1)
        self._init_weight()

    def forward(self, feature_rgb, feature_depth, mask):
        low_level_feature_rgb = self.project_rgb( feature_rgb['low_level'] ) if feature_rgb is not None else None
        low_level_feature_depth = self.project_depth( feature_depth['low_level'] ) if feature_depth is not None else None

        if low_level_feature_rgb is not None:
            low_expand_rgb = torch.zeros(mask.shape[0], low_level_feature_rgb.shape[1], low_level_feature_rgb.shape[2], low_level_feature_rgb.shape[3]).to(low_level_feature_rgb.device)
            low_expand_rgb[torch.where(mask[:, 0] != 0)[0]] = low_level_feature_rgb
        else:
            low_expand_rgb = 0
        
        if low_level_feature_depth is not None:
            low_expand_depth = torch.zeros(mask.shape[0], low_level_feature_depth.shape[1], low_level_feature_depth.shape[2], low_level_feature_depth.shape[3]).to(low_level_feature_depth.device)
            low_expand_depth[torch.where(mask[:, 1] != 0)[0]] = low_level_feature_depth
        else:
            low_expand_depth = 0
        
        low_feature_common = low_expand_rgb + low_expand_depth
        low_trans_rgb = self.low_transition_rgb(low_feature_common) + low_expand_rgb
        low_trans_depth = self.low_transition_depth(low_feature_common) + low_expand_depth

        # output_feature = self.aspp(feature['out'])
        aspp_rgb = self.aspp_rgb(feature_rgb['out']) if feature_rgb is not None else None
        aspp_depth = self.aspp_depth(feature_depth['out']) if feature_depth is not None else None

        if aspp_rgb is not None:
            expand_rgb = torch.zeros(mask.shape[0], aspp_rgb.shape[1], aspp_rgb.shape[2], aspp_rgb.shape[3]).to(aspp_rgb.device)
            expand_rgb[torch.where(mask[:, 0] != 0)[0]] = aspp_rgb
        else:
            expand_rgb = 0

        if aspp_depth is not None:
            expand_depth = torch.zeros(mask.shape[0], aspp_depth.shape[1], aspp_depth.shape[2], aspp_depth.shape[3]).to(aspp_depth.device)
            expand_depth[torch.where(mask[:, 1] != 0)[0]] = aspp_depth
        else:
            expand_depth = 0

        # print(low_expand_depth.shape)
        # print(expand_depth.shape)

        # output_feature_rgb = F.interpolate(expand_rgb, size=low_level_feature_rgb.shape[2:], mode='bilinear', align_corners=False)
        # output_feature_depth = F.interpolate(expand_depth, size=low_level_feature_depth.shape[2:], mode='bilinear', align_corners=False)
        
        # output_rgb = torch.cat([output_feature_rgb, low_trans_rgb], dim=1)
        # output_depth = torch.cat([output_feature_depth, low_trans_depth], dim=1)

        feature_common = expand_rgb + expand_depth
        trans_rgb = self.transition_rgb(feature_common) + expand_rgb
        trans_depth = self.transition_depth(feature_common) + expand_depth
        # trans_rgb = self.transition_rgb(feature_common)
        # trans_depth = self.transition_depth(feature_common)
        trans_cat = torch.cat([trans_rgb, trans_depth], dim=1)

        low_trans_cat = torch.cat([low_trans_rgb, low_trans_depth], dim=1)

        # output_feature_rgb = F.interpolate(aspp_rgb, size=low_level_feature_rgb.shape[2:], mode='bilinear', align_corners=False)
        # output_feature_depth = F.interpolate(aspp_depth, size=low_level_feature_depth.shape[2:], mode='bilinear', align_corners=False)




        # return self.classifier( torch.cat( [ low_level_feature, output_feature ], dim=1 ) )

        out1 = self.classifier(trans_cat)
        out_out1 = F.interpolate(out1, size=low_trans_cat.shape[2:], mode='bilinear', align_corners=False)
        # print(out1.shape, low_trans_cat.shape)
        out1 = torch.cat([low_trans_cat, out_out1], dim=1)
        out2 = self.fianl(out1)
        
        # return self.classifier(trans_cat)
        return out2
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHead, self).__init__()

        self.ASPP_rgb = ASPP(in_channels, aspp_dilate)
        self.ASPP_depth = ASPP(in_channels, aspp_dilate)

        # self.self_att_rgb = nn.MultiheadAttention(256, 4)
        # self.self_att_depth = nn.MultiheadAttention(256, 4)
        # self.self_att_full_rgb = nn.MultiheadAttention(256, 4)
        # self.self_att_full_depth = nn.MultiheadAttention(256, 4)
        # # print(self.self_att_rgb.out_proj.weight)
        # self.dropout1 = nn.Dropout(0.1)
        # self.dropout2 = nn.Dropout(0.1)
        # self.dropout3 = nn.Dropout(0.1)
        # self.dropout4 = nn.Dropout(0.1)
        # self.norm1 = nn.LayerNorm(256)
        # self.norm2 = nn.LayerNorm(256)
        # self.norm3 = nn.LayerNorm(256)
        # self.norm4 = nn.LayerNorm(256)
        # self.maxpool1 = nn.MaxPool2d((10, 20))
        # self.maxpool2 = nn.MaxPool2d((10, 20))
        # self.kernel_rgb = nn.Conv2d(256, 256, 5, padding=2, bias=False)
        # self.kernel_depth = nn.Conv2d(256, 256, 5, padding=2, bias=False)
        # self.kernel_rgb = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        # self.kernel_depth = nn.Conv2d(256, 256, 3, padding=1, bias=False)

        # self.private_rgb = nn.Sequential(
        #     nn.Conv2d(256, 256, 1, bias=False),
        #     #nn.BatchNorm2d(256),
        #     nn.Sigmoid(),
        # )
        # self.private_depth = nn.Sequential(
        #     nn.Conv2d(256, 256, 1, bias=False),
        #     #nn.BatchNorm2d(256),
        #     nn.Sigmoid(),
        # )
        # self.shared = nn.Sequential(
        #     nn.Conv2d(256, 256, 1, bias=False),
        #     #nn.BatchNorm2d(256),
        #     nn.Sigmoid(),
        #     # nn.ReLU(inplace=True)
        # )
        # self.recon_rgb = nn.Conv2d(256, 256, 1, bias=False)
        # self.recon_depth = nn.Conv2d(256, 256, 1, bias=False)
        # self.softmax = nn.Softmax(dim=1)
        # self.kernel_rgb = Conv2d_Sparse(256, 256, 3, padding=1, num_sparsity_groups=4, num_sparsity_cardinals=16, bias=False)
        # self.kernel_depth = Conv2d_Sparse(256, 256, 3, padding=1, num_sparsity_groups=4, num_sparsity_cardinals=16, bias=False)
        self.kernel_rgb = Conv2d_Sparse(256, 256, 3, reduction=16, padding=1, bias=False)
        self.kernel_depth = Conv2d_Sparse(256, 256, 3, reduction=16, padding=1, bias=False)
        # self.kernel_rgb = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        # self.kernel_depth = nn.Conv2d(256, 256, 3, padding=1, bias=False)

        # self.transition_rgb = nn.Sequential(
        #     nn.Conv2d(256, 256, 1, bias=False),
        #     nn.BatchNorm2d(256),
        #     # nn.LayerNorm((256, 24, 48)),
        #     nn.ReLU(inplace=True)
        # )
        # self.transition_depth = nn.Sequential(
        #     nn.Conv2d(256, 256, 1, bias=False),
        #     nn.BatchNorm2d(256),
        #     # nn.LayerNorm((256, 24, 48)),
        #     nn.ReLU(inplace=True)
        # )
        # self.mlp = nn.Conv2d(256, 256, 1)
        self.classifier = nn.Sequential(
            nn.Conv2d(256 * 2, 256, 3, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        # self.classifier = nn.Conv2d(256 * 2, num_classes, 1)
        self._init_weight()
        

    def forward(self, feature_rgb, feature_depth, mask):
        rgb_not_miss_idx = torch.where(mask[:, 0] != 0)[0]  ### 实际上是not missing
        depth_not_miss_idx = torch.where(mask[:, 1] != 0)[0]
        full_samples = torch.Tensor([idx for idx in rgb_not_miss_idx if idx in depth_not_miss_idx]).to(mask.device, dtype=torch.long)
        full_sample_at_depth_loc = [depth_not_miss_idx.tolist().index(idx) for idx in full_samples]
        full_sample_at_rgb_loc = [rgb_not_miss_idx.tolist().index(idx) for idx in full_samples]
        aspp_rgb = self.ASPP_rgb(feature_rgb['out']) if feature_rgb is not None else None
        aspp_depth = self.ASPP_depth(feature_depth['out']) if feature_depth is not None else None

        rgb_bn = self.ASPP_rgb.project[1]
        depth_bn = self.ASPP_depth.project[1]
        list_1 = [aspp_rgb, depth_bn]
        list_2 = [aspp_depth, rgb_bn]
        if feature_rgb is not None:
            # aspp_rgb = self.ASPP_rgb(feature_rgb['out'])
            pred_depth, sparse_depth, sim_depth = self.kernel_depth(list_1)
            bs_input_shape = (mask.shape[0], aspp_rgb.shape[1], aspp_rgb.shape[2], aspp_rgb.shape[3])
            device = aspp_rgb.device
        else:
            # aspp_rgb = None
            pred_depth = None
            sparse_depth = torch.zeros((aspp_depth.shape[0], 256), requires_grad=True).to(aspp_depth.device)

        if feature_depth is not None:
            # aspp_depth = self.ASPP_depth(feature_depth['out'])
            pred_rgb, sparse_rgb, sim_rgb = self.kernel_rgb(list_2)
            bs_input_shape = (mask.shape[0], aspp_depth.shape[1], aspp_depth.shape[2], aspp_depth.shape[3])
            device = aspp_depth.device
        else:
            # aspp_depth = None
            pred_rgb = None
            sparse_rgb = torch.zeros((aspp_rgb.shape[0], 256), requires_grad=True).to(aspp_rgb.device)
            
        

        # aspp_rgb = self.aspp_rgb(feature_rgb['out']) if feature_rgb is not None else None
        # aspp_depth = self.aspp_depth(feature_depth['out']) if feature_depth is not None else None

        # shared_depth = self.shared(aspp_depth) if aspp_depth is not None else None 
        # shared_rgb = self.shared(aspp_rgb) if aspp_rgb is not None else None

        # # pred_rgb = self.kernel_rgb(aspp_depth) if aspp_depth is not None else None
        # # pred_depth = self.kernel_depth(aspp_rgb) if aspp_rgb is not None else None

        # pred_rgb = self.kernel_rgb(shared_depth) if aspp_depth is not None else None
        # pred_depth = self.kernel_depth(shared_rgb) if aspp_rgb is not None else None
        

        
        

        # res_pred_rgb = aspp_rgb[full_sample_at_rgb_loc] + 0.1 * pred_rgb[full_sample_at_depth_loc] if aspp_rgb is not None and pred_rgb is not None else None
        # res_pred_depth = aspp_depth[full_sample_at_depth_loc] + 0.1 * pred_depth[full_sample_at_rgb_loc] if aspp_depth is not None and pred_depth is not None else None
        
        # if aspp_rgb is not None:
        #     b_rgb, c, h, w = aspp_rgb.shape
        #     rgb_reshape = aspp_rgb.flatten(2).permute(2, 0, 1)
        #     att_aspp_rgb = self.self_att_rgb(query=rgb_reshape, key=rgb_reshape, value=rgb_reshape)[0]
        #     att_aspp_rgb = rgb_reshape + self.dropout1(att_aspp_rgb)
        #     att_aspp_rgb = self.norm1(att_aspp_rgb)
        #     att_aspp_rgb = att_aspp_rgb.permute(1, 2, 0).view(b_rgb, c, h, w)

        # if aspp_depth is not None:
        #     b_depth, c, h, w = aspp_depth.shape
        #     depth_reshape = aspp_depth.flatten(2).permute(2, 0, 1)
        #     att_aspp_depth = self.self_att_depth(query=depth_reshape, key=depth_reshape, value=depth_reshape)[0]
        #     att_aspp_depth = depth_reshape + self.dropout2(att_aspp_depth)
        #     att_aspp_depth = self.norm2(att_aspp_depth)
        #     att_aspp_depth = att_aspp_depth.permute(1, 2, 0).view(b_depth, c, h, w)
        
        # if len(full_samples) != 0:
        #     b_full, c, h, w = aspp_rgb[full_sample_at_rgb_loc].shape
        #     full_rgb_reshape = aspp_rgb[full_sample_at_rgb_loc].flatten(2).permute(2, 0, 1)
        #     full_depth_reshape = aspp_depth[full_sample_at_depth_loc].flatten(2).permute(2, 0, 1)
        #     cat_full = torch.cat([full_rgb_reshape, full_depth_reshape])
        #     all_exist_att_rgb = self.self_att_full_rgb(query=full_rgb_reshape, key=cat_full, value=cat_full)[0]
        #     all_exist_att_depth = self.self_att_full_depth(query=full_depth_reshape, key=cat_full, value=cat_full)[0]
        #     all_exist_att_rgb = full_rgb_reshape + self.dropout3(all_exist_att_rgb)
        #     all_exist_att_depth = full_depth_reshape + self.dropout4(all_exist_att_depth)
        #     all_exist_att_rgb = self.norm3(all_exist_att_rgb)
        #     all_exist_att_depth = self.norm4(all_exist_att_depth)
        #     att_aspp_rgb[full_sample_at_rgb_loc] = all_exist_att_rgb.permute(1, 2, 0).view(b_full, c, h, w)
        #     att_aspp_depth[full_sample_at_depth_loc] = all_exist_att_depth.permute(1, 2, 0).view(b_full, c, h, w)

        # pred_depth = self.kernel_depth(att_aspp_rgb) if aspp_rgb is not None else None
        # pred_rgb = self.kernel_rgb(att_aspp_depth) if aspp_depth is not None else None


        if aspp_rgb is not None:
            expand_rgb = torch.zeros(bs_input_shape, requires_grad=True).to(device)
            expand_rgb[depth_not_miss_idx] = pred_rgb if aspp_depth is not None else 0
            expand_rgb[rgb_not_miss_idx] = aspp_rgb   ###############################################
        else:
            expand_rgb = torch.zeros(bs_input_shape, requires_grad=True).to(device)
            expand_rgb[depth_not_miss_idx] = pred_rgb

        if aspp_depth is not None:
            expand_depth = torch.zeros(bs_input_shape, requires_grad=True).to(device)
            expand_depth[rgb_not_miss_idx] = pred_depth if aspp_rgb is not None else 0
            expand_depth[depth_not_miss_idx] = aspp_depth    ######################################
        else:
            expand_depth = torch.zeros(bs_input_shape, requires_grad=True).to(device)
            expand_depth[rgb_not_miss_idx] = pred_depth

        if full_samples.shape[0] != 0:
            depth_pair = [pred_depth[full_sample_at_rgb_loc].to(aspp_rgb.device), expand_depth[full_samples].to(device)]
            rgb_pair = [pred_rgb[full_sample_at_depth_loc].to(aspp_depth.device), expand_rgb[full_samples].to(device)]
        else:
            depth_pair = [torch.zeros_like(expand_depth, requires_grad=True).to(device),torch.zeros_like(expand_depth, requires_grad=True).to(device)]
            rgb_pair = [torch.zeros_like(expand_rgb, requires_grad=True).to(device),torch.zeros_like(expand_rgb, requires_grad=True).to(device)]

        # if aspp_depth is not None:
            
        #     # diff_depth_pair = [private_depth, shared_depth]
        #     # recon_depth = self.recon_depth(shared_depth + private_depth)
        #     # recon_depth_pair = [recon_depth, aspp_depth]
        # else:
            
        #     # diff_depth_pair = [torch.tensor(0.0, requires_grad=True),torch.tensor(0.0, requires_grad=True)]
        #     # recon_depth_pair = [torch.tensor(0.0, requires_grad=True),torch.tensor(0.0, requires_grad=True)]

        # if aspp_rgb is not None and aspp_depth is not None:
        #     align_pair = [self.softmax(aspp_rgb[full_sample_at_rgb_loc]), self.softmax(aspp_depth[full_sample_at_depth_loc])]
        # #     sim_pair = [shared_rgb[full_sample_at_rgb_loc], shared_depth[full_sample_at_depth_loc]]
        # #     cross_diff_pair = [private_rgb[full_sample_at_rgb_loc], private_depth[full_sample_at_depth_loc]]

        # else:
        #     align_pair = [torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)]
        # #     sim_pair = [torch.tensor(0.0, requires_grad=True),torch.tensor(0.0, requires_grad=True)]
        # #     cross_diff_pair = [torch.tensor(0.0, requires_grad=True),torch.tensor(0.0, requires_grad=True)]
        # attention = nn.functional.softmax(self.mlp(expand_rgb), dim=1)
        # expand_depth = attention * expand_depth
        # expand_rgb = self.ASPP_rgb(expand_rgb)
        # expand_depth = self.ASPP_depth(expand_depth)
        feature_common = torch.cat([expand_rgb, expand_depth], dim=1)
        sparse_list = [sparse_rgb, sparse_depth]
        if full_samples.shape[0] != 0:
            sim_list = [sim_rgb[full_sample_at_depth_loc], sim_depth[full_sample_at_rgb_loc]]
        else:
            sim_list = [torch.zeros(bs_input_shape, requires_grad=True).to(expand_rgb.device), torch.zeros(bs_input_shape, requires_grad=True).to(expand_rgb.device)]
        #decoup_list = [sim_pair, cross_diff_pair, diff_rgb_pair, diff_depth_pair, recon_rgb_pair, recon_depth_pair]
        return self.classifier(feature_common), rgb_pair, depth_pair, sparse_list, sim_list #, expand_rgb, expand_depth#, align_pair#, decoup_list

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     # nn.init.xavier_normal_(m.weight)
            #     nn.init.constant_(m.weight, 1.0)
            #     nn.init.constant_(m.bias, 0)

class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                            stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d( in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=in_channels ),
            # PointWise Conv
            nn.Conv2d( in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )
        
        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(32, out_channels),
            # nn.LayerNorm((256, 24, 48)),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(32, out_channels),
            # nn.LayerNorm((256, 1, 1)),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(32, out_channels),
            # nn.LayerNorm((256, 24, 48)),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(32, out_channels),
            # nn.LayerNorm((256, 24, 48)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)



def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0]>1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                      module.out_channels, 
                                      module.kernel_size,
                                      module.stride,
                                      module.padding,
                                      module.dilation,
                                      module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module



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
    print(sparsity.shape)
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
    # self.linear_trans = nn.Linear(in_channels, in_channels)
    self.mlp = nn.Sequential(
        nn.Linear(in_channels, in_channels//reduction, bias=False),
        # nn.ReLU(inplace=True),
        nn.Linear(in_channels//reduction, out_channels, bias=False),
        # nn.Sigmoid()
    )
    self.mlp_bn = nn.Linear(in_channels, out_channels)


  def _conv_forward(self, input, weight, bias):

    coeff_input = input[0]
    # b, c = coeff_input.shape[:2]
    # sparsity = self.mlp(self.avg_pool(coeff_input).view(b,c))
    sparsity = self.mlp(coeff_input.mean([-1, -2]))
    sparsity = sparsity.abs()
    # if input[1] is not None:
    #     print(input[1].weight)
    bn_w = input[1].weight.abs()
    bn_w = self.mlp_bn(bn_w)
    bn_w = nn.functional.sigmoid(bn_w)
    sparse_input = torch.einsum("bk,bkhw->bkhw", (sparsity, coeff_input))

    weight = torch.einsum("i,ijhw->ijhw", (bn_w, weight))

    # if bias is not None:
        # bias = bn_bias + bias

    output = nn.functional.conv2d(sparse_input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    return output, sparsity, sparse_input