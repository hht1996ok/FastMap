import copy
import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.init import normal_
import torch.nn.functional as F
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from torchvision.transforms.functional import rotate
from mmdet3d.bevformer.modules.temporal_self_attention import TemporalSelfAttention
from mmdet3d.bevformer.modules.spatial_cross_attention import MSDeformableAttention3D
from mmdet3d.bevformer.modules.decoder import CustomMSDeformableAttention
from mmdet3d.models.builder import build_fuser, FUSERS
from mmdet3d.bevformer.modules import CustomInstanceSelfAttention, TransformerDecoderLayer
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv import ConfigDict
from typing import List
from .decoder import build_feedforward_network
import warnings

def clip_sigmoid(x, eps=1e-4):
    x_ = x.sigmoid()
    y = torch.clamp(x_, min=eps, max=1 - eps)
    return y


@TRANSFORMER.register_module()
class MapTRPerceptionHeatmapTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 z_cfg=dict(
                    pred_z_flag=False,
                    gt_z_flag=False,
                 ),
                 ffn_cfgs = dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=1024,
                    num_fcs=2,
                    ffn_drop=0.,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                 two_stage_num_proposals=300,
                 num_proposals=2100,
                 fuser=None,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 heatmap_weight=1.0,
                 rotate_prev_bev=True,
                 use_add=True,
                 use_attention=True,
                 attention_version='v1',
                 use_fast_perception_neck=True,
                 num_decoders=3,
                 nms_kernel_size=3,
                 num_classes=3,
                 bev_h=200,
                 bev_w=100,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 heatmap_version=None,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 modality='vision',
                 feat_down_sample_indice=-1,
                 **kwargs):
        super(MapTRPerceptionHeatmapTransformer, self).__init__(**kwargs)
        if modality == 'fusion':
            self.fuser = build_fuser(fuser) #TODO
        # self.use_attn_bev = encoder['type'] == 'BEVFormerEncoder'
        self.use_attn_bev = 'BEVFormerEncoder' in encoder['type']
        self.encoder = build_transformer_layer_sequence(encoder)
        if use_fast_perception_neck:
            self.decoder = nn.ModuleList()
            for i in range(num_decoders):
                self.decoder.append(build_transformer_layer_sequence(decoder))
        else:
            self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.use_fast_perception_neck = use_fast_perception_neck
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.heatmap_weight = heatmap_weight
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.nms_kernel_size = nms_kernel_size
        self.use_add = use_add

        self.two_stage_num_proposals = two_stage_num_proposals
        self.z_cfg=z_cfg
        self.bev_pos_2d = self.create_2D_grid(bev_h, bev_w)
        self.heatmap_version = heatmap_version
        self.use_attention = use_attention
        self.attention_version = attention_version
        self.init_layers()
        self.rotate_center = rotate_center
        self.feat_down_sample_indice = feat_down_sample_indice

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs')
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f'The arguments `{ori_name}` in BaseTransformerLayer '
                    f'has been deprecated, now you should set `{new_name}` '
                    f'and other FFN related arguments '
                    f'to a dict named `ffn_cfgs`. ')
                ffn_cfgs[new_name] = kwargs[ori_name]

        self.ffn = ModuleList()
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(1)]
        assert len(ffn_cfgs) == 1
        for ffn_index in range(1):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffn.append(
                build_feedforward_network(ffn_cfgs[ffn_index], dict(type='FFN')))

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        # self.reference_points = nn.Sequential(
        #     nn.Conv1d(self.embed_dims, self.embed_dims, kernel_size=1, padding=0),
        #     nn.BatchNorm1d(self.embed_dims),
        #     nn.ReLU(),
        #     nn.Conv1d(self.embed_dims, 2, kernel_size=1, padding=0),
        # )
        #self.query_self_att = CustomInstanceSelfAttention(embed_dims=self.embed_dims, num_heads=8, dropout=0.1)
        self.query_self_att_instances = CustomInstanceSelfAttention(embed_dims=self.embed_dims, num_heads=8, dropout=0.1)
        self.query_self_att_points = CustomInstanceSelfAttention(embed_dims=self.embed_dims, num_heads=8, dropout=0.1)
        self.query_cross_att = TransformerDecoderLayer(d_model=self.embed_dims, nhead=8, cross_only=True)
        self.key_pos_embed = nn.Conv1d(2, self.embed_dims, 1)
        self.pos_embed = nn.Conv1d(self.embed_dims, self.embed_dims, 1)
        self.class_encoding = nn.Conv1d(self.num_classes, self.embed_dims, 1)
        if self.use_attention:
            if self.attention_version == 'v1':
                self.attention = nn.Sequential(
                    nn.Conv1d(self.embed_dims * 2, self.embed_dims // 2, 1),
                    nn.BatchNorm1d(self.embed_dims // 2),
                    nn.ReLU(),
                    nn.Conv1d(self.embed_dims // 2, 1, 1),
                )
            else:
                self.attention = nn.Sequential(
                    nn.Conv1d(self.embed_dims * 2, self.embed_dims // 2, 1),
                    nn.BatchNorm1d(self.embed_dims // 2),
                    nn.ReLU(),
                    nn.Conv1d(self.embed_dims // 2, 2, 1),
                )
        # self.ffn = nn.Sequential(
        #     nn.Conv1d(self.embed_dims, self.embed_dims, kernel_size=1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv1d(self.embed_dims, self.embed_dims, kernel_size=1, padding=0)
        # )
        # self.ffn_bn = nn.BatchNorm1d(self.embed_dims)

        # layers = []
    #     layers.append(
    #         ConvModule(
    #             self.embed_dims,
    #             self.embed_dims,
    #             kernel_size=3,
    #             padding=1,
    #             # bias=bias,
    #             conv_cfg=dict(type="Conv2d"),
    #             norm_cfg=dict(type="BN2d"),
    #         )
    #     )
        # layers.append(
        #     build_conv_layer(
        #         dict(type="Conv2d"),
        #         self.embed_dims,
        #         self.num_classes,
        #         kernel_size=3,
        #         padding=1,
        #         # bias=bias,
        #     )
        # )
        # self.heatmap_head = nn.Sequential(*layers)
        #self.heatmap_head = UNet(in_channels=self.embed_dims, num_classes=self.num_classes)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.embed_dims * 2),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims * 2, self.embed_dims * 4, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(self.embed_dims * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(self.embed_dims * 4, self.embed_dims * 2, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(self.embed_dims * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.embed_dims * 2, self.embed_dims, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.BatchNorm2d(self.embed_dims),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims, 3, kernel_size=1, padding=0),
        )
        #self.heatmap_head = BEVMaskGenerator(bev_channels=256, classes=self.num_classes)

    def gaussian_2d(self, shape, sigma):
        """Generate gaussian map.

        Args:
            shape (list[int]): Shape of the map.
            sigma (float): Sigma to generate gaussian map.
                Defaults to 1.

        Returns:
            np.ndarray: Generated gaussian map.
        """
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = 1.0 - np.exp(-(x * x + y * y) / (2 * sigma[0] * sigma[1]))
        return h

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        #xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def lss_bev_encode(
            self,
            mlvl_feats,
            prev_bev=None,
            **kwargs):
        # import ipdb;ipdb.set_trace()
        # assert len(mlvl_feats) == 1, 'Currently we only use last single level feat in LSS'
        # import ipdb;ipdb.set_trace()
        images = mlvl_feats[self.feat_down_sample_indice]
        img_metas = kwargs['img_metas']
        encoder_outputdict = self.encoder(images,img_metas)
        bev_embed = encoder_outputdict['bev']
        depth = encoder_outputdict['depth']
        if 'depth_pre' in encoder_outputdict:
            depth_pre = encoder_outputdict['depth_pre']
            bs, c, _,_ = bev_embed.shape
            bev_embed = bev_embed.view(bs,c,-1).permute(0,2,1).contiguous()
            ret_dict = dict(
                bev=bev_embed,
                depth=depth,
                depth_pre=depth_pre,
            )
        else:
            bs, c, _,_ = bev_embed.shape
            #bev_embed = bev_embed.view(bs,c,-1).permute(0,2,1).contiguous()
            ret_dict = dict(
                bev=bev_embed,
                depth=depth,
            )
        return ret_dict

    def get_bev_features(
            self,
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """
        ret_dict = self.lss_bev_encode(
            mlvl_feats,
            prev_bev=prev_bev,
            **kwargs)
        bev_embed = ret_dict['bev']
        depth = ret_dict['depth']
        if 'depth_pre' in ret_dict:
            depth_pre = ret_dict['depth_pre']
            if lidar_feat is not None:
                bs = mlvl_feats[0].size(0)
                bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0,3,1,2).contiguous()
                lidar_feat = lidar_feat.permute(0,1,3,2).contiguous() # B C H W
                lidar_feat = nn.functional.interpolate(lidar_feat, size=(bev_h, bev_w), mode='bicubic', align_corners=False)
                bev_embed = self.fuser([bev_embed, lidar_feat])
            else:
                bs = mlvl_feats[0].size(0)
                bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0,3,1,2).contiguous()
            ret_dict = dict(
                bev=bev_embed,
                depth=depth,
                depth_pre=depth_pre
            )
        else:
            if lidar_feat is not None:
                bs = mlvl_feats[0].size(0)
                bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0,3,1,2).contiguous()
                lidar_feat = lidar_feat.permute(0,1,3,2).contiguous() # B C H W
                lidar_feat = nn.functional.interpolate(lidar_feat, size=(bev_h, bev_w), mode='bicubic', align_corners=False)
                bev_embed = self.fuser([bev_embed, lidar_feat])
            else:
                bs = mlvl_feats[0].size(0)
                #bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0,3,1,2).contiguous()
            ret_dict = dict(
                bev=bev_embed,
                depth=depth,
            )
        return ret_dict

    def format_feats(self, mlvl_feats):
        bs = mlvl_feats[0].size(0)
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)

            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        return feat_flatten, spatial_shapes, level_start_index

    def extract_heatmap_indices_and_classes(self, heatmap, n, m):
        """
        从heatmap中按n为大小分区域提取每个区域中置信度最大的m个像素的索引和类别

        参数：
        heatmap：输入的heatmap，尺寸为(b, 3, h, w)，类型为torch.Tensor
        n：每个区域的大小
        m：每个区域中需要提取的像素数量

        返回值：
        extracted_indices：提取的像素索引，尺寸为(b, num_regions, m)，num_regions为区域数量
        extracted_classes：提取的像素类别，尺寸为(b, num_regions, m)
        """
        b, c, h, w = heatmap.shape
        # 计算每个维度可以分成的区域数量
        num_regions_h = h // n
        num_regions_w = w // n
        num_regions = num_regions_h * num_regions_w

        # 将heatmap按区域大小展开
        unfolded_heatmap = heatmap.unfold(2, n, n).unfold(3, n, n)
        # 调整维度顺序，将类别维度和区域维度放到前面
        unfolded_heatmap = unfolded_heatmap.permute(0, 2, 3, 1, 4, 5).contiguous()
        # 将每个区域的heatmap展平
        unfolded_heatmap = unfolded_heatmap.view(b, num_regions, -1)

        # 获取每个区域中置信度最大的m个像素的索引
        top_m_values, top_m_indices = torch.topk(unfolded_heatmap, m, dim=2)
        # 获取每个区域中置信度最大的m个像素的类别
        top_m_classes = top_m_indices // (n * n)
        # 获取每个区域中置信度最大的m个像素在原heatmap中的索引
        top_m_indices = top_m_indices % (n * n)
        top_m_indices = top_m_indices + torch.arange(0, num_regions * n * n, n * n).view(1, -1, 1).to(top_m_indices.device)
        top_m_indices = top_m_indices.reshape(b, -1)
        top_m_classes = top_m_classes.reshape(b, -1)

        return top_m_indices, top_m_classes

    def select_top_k_points(self, mask, m=250, num_regions=10):
        """
        输入:
            mask: 形状为(batch, h, w, 3)的 mask，其中 3代表类别。
            m: 每个区域选择的点数，默认为 10 。
            num_regions: 划分的圆形区域数量，默认为 10 。
        输出:
            selected_indices: 形状为(batch, num_regions, m)的张量，表示每个区域中置信度最大的 m个像素的索引。
            selected_classes: 形状为(batch, num_regions, m)的张量，表示每个区域中置信度最大的 m个像素的类别。
        """
        mask = mask.permute(0, 2, 3, 1)
        batch_size, h, w, num_classes = mask.shape
        # 计算中心点坐标
        center_y, center_x = h // 2, w // 2
        # 生成网格坐标
        y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y_coords = y_coords.float().to(mask.device)
        x_coords = x_coords.float().to(mask.device)

        # 计算每个像素点到中心点的距离
        distances = torch.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
        # 将 mask 展平为(batch, h*w, num_classes)
        flat_mask = mask.reshape(batch_size, -1, num_classes)
        flat_distances = distances.reshape(-1)
        # 划分圆形区域
        max_distance = flat_distances.max()
        region_width = max_distance / num_regions
        region_indices = (flat_distances // region_width).clamp(max=num_regions-1).long()

        # 计算每个区域的面积比例
        region_areas = [math.sqrt(i+1) for i in range(num_regions)]  # 面积与半径平方成正比
        total_area = sum(region_areas)
        region_points = [int(m * area / total_area) for area in region_areas]  # 按面积比例分配采样点数

        # 初始化结果张量
        max_points = max(region_points)  # 每个区域的最大采样点数
        selected_indices = torch.zeros(batch_size, num_regions, max_points, dtype=torch.long, device=mask.device)
        selected_classes = torch.zeros(batch_size, num_regions, max_points, dtype=torch.long, device=mask.device)

        # 遍历每个区域
        for region_idx in range(num_regions):
            # 创建区域掩码
            region_mask = (region_indices == region_idx)
            # 获取当前区域的 mask 值和类别
            region_mask_values = flat_mask[:, region_mask, :]  # 形状为(batch, region_size, num_classes)
            region_size = region_mask_values.shape[1]
            # 当前区域的采样点数
            current_m = region_points[region_idx]
            # 如果区域内的点少于 current_m 个，则选择所有点
            if region_size <= current_m:
                top_k_indices = torch.arange(region_size, device=mask.device).expand(batch_size, -1)
            else:
                # 选择置信度最高的 current_m 个点
                _, top_k_indices = torch.topk(region_mask_values.max(dim=-1).values, current_m, dim=1)
            # 获取选中的点的索引和类别
            flat_indices = torch.nonzero(region_mask).squeeze(1)
            selected_flat_indices = flat_indices[top_k_indices]  # 形状为(batch, current_m)
            selected_indices[:, region_idx, :current_m] = selected_flat_indices
            # 获取选中的点的类别
            selected_class_values = region_mask_values.gather(1, top_k_indices.unsqueeze(-1).expand(-1, -1, num_classes))
            selected_classes[:, region_idx, :current_m] = selected_class_values.argmax(dim=-1)

        selected_indices = selected_indices.reshape(batch_size, -1)
        selected_classes = selected_classes.reshape(batch_size, -1)
        return selected_indices, selected_classes

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    @force_fp32(apply_to=('mlvl_feats', 'lidar_feat'))
    def forward(self,
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                backbone=None,
                neck=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """
        ouput_dic = self.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bev_embed = ouput_dic['bev']
        depth = ouput_dic['depth']
        if 'depth_pre' in ouput_dic:
            depth_pre = ouput_dic['depth_pre']
        if self.use_fast_perception_neck:
            depth_list = []
            inter_states_list = []
            init_reference_out_list = []
            inter_references_out_list = []
            bev_embeds = neck(bev_embed) # list 上采样分支数量

            for index, bev_embed in enumerate(bev_embeds):
                bev_embed = bev_embed.flatten(2).permute(0, 2, 1).contiguous().float()
                bs = mlvl_feats[0].size(0)
                query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
                query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
                query = query.unsqueeze(0).expand(bs, -1, -1)
                reference_points = self.reference_points(query_pos)
                reference_points = reference_points.sigmoid()
                init_reference_out = reference_points

                query = query.permute(1, 0, 2)
                query_pos = query_pos.permute(1, 0, 2)
                bev_embed = bev_embed.permute(1, 0, 2)

                feat_flatten, feat_spatial_shapes, feat_level_start_index = self.format_feats(mlvl_feats)

                inter_states, inter_references = self.decoder[index](
                    query=query,
                    key=None,
                    value=bev_embed,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    reg_branches=reg_branches,
                    cls_branches=cls_branches,
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    mlvl_feats=mlvl_feats,
                    feat_flatten=feat_flatten,
                    feat_spatial_shapes=feat_spatial_shapes,
                    feat_level_start_index=feat_level_start_index,
                    **kwargs)

                inter_references_out = inter_references
                inter_states_list.append(inter_states)
                init_reference_out_list.append(init_reference_out)
                inter_references_out_list.append(inter_references_out)
            return bev_embeds, depth, inter_states_list, init_reference_out_list, inter_references_out_list  # depth不需要list
        else:
            if backbone is not None:
                bev_embed = neck(backbone(bev_embed))[0].float()
            bev_embed_ = bev_embed.flatten(2).permute(0, 2, 1).contiguous().float().permute(1, 0, 2)
            depth = ouput_dic['depth']
            bs = mlvl_feats[0].size(0)
            query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1).permute(1, 0, 2)
            query = query.unsqueeze(0).expand(bs, -1, -1).permute(1, 0, 2)

            bev_pos_2d = self.bev_pos_2d.repeat(bs, 1, 1).to(bev_embed.device)
            dense_heatmap = self.heatmap_head(bev_embed)

            heatmap = clip_sigmoid(dense_heatmap).detach()
            if self.heatmap_version == 'v1':
                heatmap_gaussian = torch.tensor(self.gaussian_2d((heatmap.size(2), heatmap.size(3)), sigma=(heatmap.size(2) / 15, heatmap.size(3) / 15))).cuda()
                heatmap_weight = heatmap_gaussian * self.heatmap_weight + 1.0
                heatmap *= heatmap_weight[None, None]
                heatmap = heatmap.reshape(bs, heatmap.shape[1], -1)
                top_proposals = heatmap.reshape(bs, -1).argsort(dim=-1, descending=True)[..., : self.num_proposals]
                top_proposals_class = top_proposals // heatmap.shape[-1]
                top_proposals_index = top_proposals % heatmap.shape[-1]
            elif self.heatmap_version == 'v2':
                top_proposals_index, top_proposals_class = self.extract_heatmap_indices_and_classes(heatmap, 5, 3)
            elif self.heatmap_version == 'v3':
                top_proposals_index, top_proposals_class = self.select_top_k_points(heatmap, self.num_proposals, 3)
            # top_proposals_div = heatmap[:, 0, :].argsort(dim=-1, descending=True)[..., : self.num_proposals // 3]
            # top_proposals_ped = heatmap[:, 1, :].argsort(dim=-1, descending=True)[..., : self.num_proposals // 3]
            # top_proposals_bou = heatmap[:, 2, :].argsort(dim=-1, descending=True)[..., : self.num_proposals // 3]
            # top_proposals = torch.cat([top_proposals_div, top_proposals_ped, top_proposals_bou], dim=1)

            bev_embed_flatten = bev_embed.view(bs, bev_embed.shape[1], -1)
            key_feat = bev_embed_flatten.gather(
                index=top_proposals_index[:, None, :].expand(
                    -1, bev_embed_flatten.shape[1], -1
                ), dim=-1,
            )
            key_pos = bev_pos_2d.gather(
                index=top_proposals_index[:, None, :]
                .permute(0, 2, 1)
                .expand(-1, -1, bev_pos_2d.shape[-1]),
                dim=1,
            ).permute(0, 2, 1)

            # pos_heatmap = torch.zeros(bev_h, bev_w).cuda()
            # for pos in key_pos[0].permute(1, 0):
            #     #pos_heatmap[pos[0].long(), pos[1].long()] = 255
            #     pos_heatmap[bev_h - pos[0].long() - 1, pos[1].long()] = 255
            #     heatmap_tmp = torch.flip(heatmap[0].max(dim=0)[0], dims=[0])
            # import cv2
            # import time
            # t1=time.time()
            # cv2.imwrite(f'./pos_weight/pos_heatmap_weight_{t1}.jpg', pos_heatmap.detach().cpu().numpy())
            # cv2.imwrite(f'./heatmap/heatmap_{t1}.jpg', heatmap_tmp.detach().cpu().numpy()*255)
            key_pos = self.key_pos_embed(key_pos)

            # self_query = self.query_self_att(
            #     query=query, #(150,4,256)
            #     key=query,
            #     value=query,
            #     query_pos=query_pos,
            #     attn_mask=kwargs['self_attn_mask'],
            # )  # 150,b,c

            num_vec_points, n_batch, n_dim = query.size()
            num_pts_per_vec = 20
            num_vec = num_vec_points // num_pts_per_vec
            query = query.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(1, 2)
            query_pos = query_pos.view(num_vec, num_pts_per_vec, n_batch, n_dim).flatten(1, 2)
            query = self.query_self_att_instances(
                query=query,
                key=query,
                value=query,
                query_pos=query_pos,
                attn_mask=kwargs['self_attn_mask'],
                **kwargs)
            query = query.view(num_vec, num_pts_per_vec, n_batch, n_dim).permute(1, 0, 2, 3).contiguous().flatten(1, 2)
            query_pos = query_pos.view(num_vec, num_pts_per_vec, n_batch, n_dim).permute(1, 0, 2, 3).contiguous().flatten(1, 2)

            self_query = self.query_self_att_points(
                query=query,
                key=query,
                value=query,
                query_pos=query_pos,
                **kwargs)
            self_query = self_query.view(num_pts_per_vec, num_vec, n_batch, n_dim).permute(1,0,2,3).contiguous().flatten(0,1)
            query_pos = query_pos.view(num_pts_per_vec, num_vec, n_batch, n_dim).permute(1,0,2,3).contiguous().flatten(0,1)

            # add category embedding
            one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(
                0, 2, 1
            )  #(B,C,num_proposals)
            query_cat_encoding = self.class_encoding(one_hot.float())
            key_feat += query_cat_encoding #(B,C,num_proposals)

            query_feat = self.query_cross_att(
                query=self_query,
                key=key_feat.permute(2, 0, 1),
                query_pos=query_pos,
                key_pos=key_pos.permute(2, 0, 1),
            ).permute(2, 0, 1)
            identity = query_feat
            query_feat = self.ffn[0](query_feat, identity)

            if self.use_attention:
                if self.attention_version == 'v1':
                    query_feat_tmp = torch.cat([query_feat, self_query], dim=2).reshape(num_vec, num_pts_per_vec, n_batch, n_dim * 2)
                    query_feat_tmp_max = query_feat_tmp.max(dim=1)[0]
                    cross_weight = self.attention(query_feat_tmp_max.permute(0, 2, 1)).sigmoid().permute(0, 2, 1)
                    cross_weight = cross_weight.unsqueeze(1).repeat(1, num_pts_per_vec, 1, 1).reshape(-1, n_batch, 1)
                    #cross_weight = self.attention(torch.cat([query_feat, self_query], dim=2).permute(0, 2, 1)).sigmoid().permute(0, 2, 1)
                    query_feat = query_feat * cross_weight + self_query * (1 - cross_weight)
                else:
                    cross_weight = self.attention(torch.cat([query_feat, self_query], dim=2).permute(0, 2, 1)).softmax(dim=1).permute(0, 2, 1)
                    query_feat = query_feat * cross_weight[:, :, :1] + self_query * cross_weight[:, :, 1:2]

            #feat_flatten, feat_spatial_shapes, feat_level_start_index = self.format_feats(mlvl_feats)
            #import pdb; pdb.set_trace()
            # query_pos = self.pos_embed((query_feat+self_query).permute(1, 2 ,0)).permute(2, 0, 1)
            # reference_points = reg_branches[0]((query_feat+self_query).permute(1, 0, 2))
            # reference_points = reference_points.sigmoid()
            # query_feat_tmp = (query_feat+self_query).unsqueeze(0)
            query_pos = self.pos_embed(query_feat.permute(1, 2 ,0)).permute(2, 0, 1)
            reference_points = reg_branches[0](query_feat.permute(1, 0, 2))
            reference_points = reference_points.sigmoid()
            query_feat_tmp = query_feat.unsqueeze(0)

            inter_states, inter_references = self.decoder(
                query=query_feat,
                key=None,
                value=bev_embed_,
                query_pos=query_pos,
                reference_points=reference_points,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                level_start_index=torch.tensor([0], device=query.device),
                mlvl_feats=mlvl_feats,
                # feat_flatten=feat_flatten,
                # feat_spatial_shapes=feat_spatial_shapes,
                # feat_level_start_index=feat_level_start_index,
                **kwargs)
            inter_references_out = inter_references
            if self.use_add:
                inter_states = inter_states + query_feat
            inter_states = torch.cat([query_feat_tmp, inter_states], dim=0)

            if 'depth_pre' in ouput_dic:
                return bev_embed, depth, inter_states, inter_references_out, dense_heatmap, depth_pre
            else:
                return bev_embed, depth, inter_states, inter_references_out, dense_heatmap
