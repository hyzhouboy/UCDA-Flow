import torch.nn as nn
import torch.nn.functional as F

from depth_nets.feature import (StereoNetFeature, PSMNetFeature, GANetFeature, GCNetFeature,
                          FeaturePyrmaid, FeaturePyramidNetwork)
from depth_nets.resnet import AANetFeature
from depth_nets.cost import CostVolume, CostVolumePyramid
from depth_nets.aggregation import (StereoNetAggregation, GCNetAggregation, PSMNetBasicAggregation,
                              PSMNetHGAggregation, AdaptiveAggregation)
from depth_nets.estimation import DisparityEstimation
from depth_nets.refinement import StereoNetRefinement, StereoDRNetRefinement, HourglassRefinement


class AANet(nn.Module):
    def __init__(self, args):
        super(AANet, self).__init__()

        self.num_downsample = args.num_downsample
        self.num_scales = args.num_scales
        self.num_fusions = 6
        self.deformable_groups=2
        self.mdconv_dilation = 2
        self.num_stage_blocks = 1
        self.num_deform_blocks = 3
        self.feature_similarity = 'correlation'
        self.no_feature_mdconv = False
        self.no_intermediate_supervision = True

        # Feature extractor
       
        self.feature_extractor = AANetFeature(feature_mdconv=(not self.no_feature_mdconv))
        self.max_disp = args.max_disp // 3
        in_channels = [32 * 4, 32 * 8, 32 * 16, ]
        
        self.fpn = FeaturePyramidNetwork(in_channels=in_channels,
                                             out_channels=32 * 4)
        # Cost volume construction
        cost_volume_module = CostVolumePyramid
        self.cost_volume = cost_volume_module(self.max_disp,
                                              feature_similarity=self.feature_similarity)

        # Cost aggregation
        max_disp = self.max_disp

        in_channels = 32  # StereoNet uses feature difference

        self.aggregation = AdaptiveAggregation(max_disp=max_disp,
                                                num_scales=self.num_scales,
                                                num_fusions=self.num_fusions,
                                                num_stage_blocks=self.num_stage_blocks,
                                                num_deform_blocks=self.num_deform_blocks,
                                                mdconv_dilation=self.mdconv_dilation,
                                                deformable_groups=self.deformable_groups,
                                                intermediate_supervision=not self.no_intermediate_supervision)

        match_similarity = False if self.feature_similarity in ['difference', 'concat'] else True

        # Disparity estimation
        self.disparity_estimation = DisparityEstimation(max_disp, match_similarity)

        # Refinement
        refine_module_list = nn.ModuleList()
        for i in range(self.num_downsample):
            refine_module_list.append(StereoDRNetRefinement())
        self.refinement = refine_module_list

    def feature_extraction(self, img):
        feature = self.feature_extractor(img)
        # if self.feature_pyramid_network or self.feature_pyramid:
        feature = self.fpn(feature)
        return feature

    def cost_volume_construction(self, left_feature, right_feature):
        cost_volume = self.cost_volume(left_feature, right_feature)

        if isinstance(cost_volume, list):
            if self.num_scales == 1:
                cost_volume = [cost_volume[0]]  # ablation purpose for 1 scale only
        else:
            cost_volume = [cost_volume]
        return cost_volume


    def disparity_computation(self, aggregation):
        if isinstance(aggregation, list):
            disparity_pyramid = []
            length = len(aggregation)  # D/3, D/6, D/12
            for i in range(length):
                disp = self.disparity_estimation(aggregation[length - 1 - i])  # reverse
                disparity_pyramid.append(disp)  # D/12, D/6, D/3
        else:
            disparity = self.disparity_estimation(aggregation)
            disparity_pyramid = [disparity]

        return disparity_pyramid


    def disparity_refinement(self, left_img, right_img, disparity):
        disparity_pyramid = []
        for i in range(self.num_downsample):
            scale_factor = 1. / pow(2, self.num_downsample - i - 1)

            if scale_factor == 1.0:
                curr_left_img = left_img
                curr_right_img = right_img
            else:
                curr_left_img = F.interpolate(left_img,
                                                scale_factor=scale_factor,
                                                mode='bilinear', align_corners=False)
                curr_right_img = F.interpolate(right_img,
                                                scale_factor=scale_factor,
                                                mode='bilinear', align_corners=False)
            inputs = (disparity, curr_left_img, curr_right_img)
            disparity = self.refinement[i](*inputs)
            disparity_pyramid.append(disparity)  # [H/2, H]

        return disparity_pyramid

    def forward(self, left_img, right_img):
        left_feature = self.feature_extraction(left_img)
        right_feature = self.feature_extraction(right_img)
        cost_volume = self.cost_volume_construction(left_feature, right_feature)
        aggregation = self.aggregation(cost_volume)
        disparity_pyramid = self.disparity_computation(aggregation)
        disparity_pyramid += self.disparity_refinement(left_img, right_img,
                                                       disparity_pyramid[-1])

        return disparity_pyramid
