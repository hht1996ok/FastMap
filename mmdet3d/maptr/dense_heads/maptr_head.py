import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet3d.models import builder

def normalize_2d_bbox(bboxes, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[...,0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[...,1:2] = cxcywh_bboxes[...,1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes

def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

def denormalize_3d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    new_pts[...,2:3] = (pts[...,2:3]*(pc_range[5] -
                            pc_range[2]) + pc_range[2])
    return new_pts

def normalize_3d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    patch_z = pc_range[5]-pc_range[2]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    new_pts[...,2:3] = pts[...,2:3] - pc_range[2]
    factor = pts.new_tensor([patch_w, patch_h,patch_z])
    normalized_pts = new_pts / factor
    return normalized_pts

def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])

    return bboxes

def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    return new_pts

def clip_sigmoid(x, eps=1e-4):
    x_ = x.sigmoid()
    y = torch.clamp(x_, min=eps, max=1 - eps)
    return y


@HEADS.register_module()
class MapTRHead(DETRHead):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 backbone=None,
                 neck=None,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 num_vec_one2one=50,
                 kernel_size=3,
                 multi_loss=False,
                 use_sigmoid=True,
                 use_coords_diff=False,
                 seg_line_coord_loss=False,
                 heatmap_loss_stop_epoch=24,
                 use_start_pts=False,
                 coords_version='v1',
                 k_one2many=0,
                 multi_loss_weight=[1.0, 1.0],
                 lambda_one2many=1,
                 num_vec_one2many=0,
                 num_pts_per_vec=2,
                 num_pts_per_gt_vec=2,
                 query_embed_type='all_pts',
                 transform_method='minmax',
                 gt_shift_pts_pattern='v0',
                 dir_interval=1,
                 use_fast_perception_neck=False,
                 loss_pts=dict(type='ChamferDistance',
                             loss_src_weight=1.0,
                             loss_dst_weight=1.0),
                 loss_coords=dict(type='ChamferDistance',
                             loss_src_weight=1.0,
                             loss_dst_weight=1.0),
                 loss_diff_pts=dict(type='ChamferDistance',
                             loss_src_weight=1.0,
                             loss_dst_weight=1.0),
                 loss_seg=dict(type='SimpleLoss',
                              pos_weight=2.13,
                              loss_weight=1.0),
                 loss_pv_seg=dict(type='SimpleLoss',
                              pos_weight=2.13,
                              loss_weight=1.0),
                 loss_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                 loss_coords_diff=dict(type='PtsL1Loss', loss_weight=0.15),
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
                 aux_seg = dict(
                    use_aux_seg=False,
                    bev_seg=False,
                    pv_seg=False,
                    seg_classes=1,
                    feat_down_sample=32,
                ),
                z_cfg = dict(
                    pred_z_flag=False,
                    gt_z_flag=False,
                ),
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.use_fast_perception_neck = use_fast_perception_neck
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.bev_encoder_type = transformer.encoder.type
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.z_cfg = z_cfg
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        self.num_vec_one2one = num_vec_one2one
        self.num_vec_one2many = num_vec_one2many
        num_vec = num_vec_one2one + num_vec_one2many
        self.kernel_size = kernel_size
        self.use_coords_diff = use_coords_diff
        self.seg_line_coord_loss = seg_line_coord_loss
        self.coords_version = coords_version
        self.use_sigmoid=use_sigmoid
        self.heatmap_loss_stop_epoch = heatmap_loss_stop_epoch
        self.use_start_pts = use_start_pts

        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.multi_loss = multi_loss
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many
        self.aux_seg = aux_seg
        self.multi_loss_weight = multi_loss_weight

        super(MapTRHead, self).__init__(
            *args, transformer=transformer, **kwargs)

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.loss_pts = build_loss(loss_pts)
        self.loss_coords = build_loss(loss_coords)
        self.loss_coords_diff = build_loss(loss_coords_diff)
        self.loss_diff_pts = build_loss(loss_diff_pts)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.loss_dir = build_loss(loss_dir)
        if backbone:
            self.backbone = builder.build_backbone(backbone)
        else:
            self.backbone = None
        if neck:
            self.neck = builder.build_neck(neck)
        else:
            self.neck = None

        self.loss_seg = build_loss(loss_seg)
        self.loss_pv_seg = build_loss(loss_pv_seg)
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        # cls_branch.append(Linear(self.embed_dims * 2, self.embed_dims))
        # cls_branch.append(nn.LayerNorm(self.embed_dims))
        # cls_branch.append(nn.ReLU(inplace=True))
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        if self.multi_loss:
            num_pred = (self.transformer.decoder.num_layers + 1)
        else:
            num_pred = self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if self.aux_seg['use_aux_seg']:
            if not (self.aux_seg['bev_seg'] or self.aux_seg['pv_seg']):
                raise ValueError('aux_seg must have bev_seg or pv_seg')
            if self.aux_seg['bev_seg']:
                self.seg_head = nn.Sequential(
                    nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
                    # nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.embed_dims, self.aux_seg['seg_classes'], kernel_size=1, padding=0)
                )
            if self.aux_seg['pv_seg']:
                self.pv_seg_head = nn.Sequential(
                    nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, padding=1, bias=False),
                    # nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.embed_dims, self.aux_seg['seg_classes'], kernel_size=1, padding=0)
                )

        if not self.as_two_stage:
            if self.bev_encoder_type == 'BEVFormerEncoder':
                self.bev_embedding = nn.Embedding(
                    self.bev_h * self.bev_w, self.embed_dims)
            else:
                self.bev_embedding = None
            if self.query_embed_type == 'all_pts':
                self.query_embedding = nn.Embedding(self.num_query,
                                                    self.embed_dims * 2)
            elif self.query_embed_type == 'instance_pts':
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
        # for m in self.reg_branches:
        #     constant_init(m[-1], 0, bias=0)
        # nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], 0.)

    # @auto_fp16(apply_to=('mlvl_feats'))

    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None, only_bev=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        if self.training:
            num_vec = self.num_vec
        else:
            num_vec = self.num_vec_one2one

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        if self.query_embed_type == 'all_pts':
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == 'instance_pts':
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight[0 : num_vec].unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)
        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight.to(dtype)
            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None

        # self_attn_mask = (
        #     torch.zeros([num_vec * self.num_pts_per_vec, num_vec * self.num_pts_per_vec,]).bool().to(mlvl_feats[0].device)
        # )
        # self_attn_mask[self.num_vec_one2one * self.num_pts_per_vec :, 0 : self.num_vec_one2one*self.num_pts_per_vec,] = True
        # self_attn_mask[0:self.num_vec_one2one * self.num_pts_per_vec, self.num_vec_one2one * self.num_pts_per_vec:,] = True

        self_attn_mask = (
            torch.zeros([num_vec, num_vec,]).bool().to(mlvl_feats[0].device)
        )
        self_attn_mask[self.num_vec_one2one :, 0 : self.num_vec_one2one,] = True
        self_attn_mask[0 : self.num_vec_one2one, self.num_vec_one2one :,] = True

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                backbone=self.backbone,
                neck=self.neck,
                img_metas=img_metas,
                prev_bev=prev_bev,
                self_attn_mask=self_attn_mask,
                num_vec=num_vec,
                num_pts_per_vec=self.num_pts_per_vec,
        )
        if self.use_fast_perception_neck:
            bev_embed_list, depth, hs_list, init_reference_list, inter_references_list, heatmap_list = outputs
            outputs_classes_list = []
            outputs_coords_list = []
            outputs_pts_coords_list = []

            outputs_classes_one2many_list = []
            outputs_coords_one2many_list = []
            outputs_pts_coords_one2many_list = []

            outputs_seg_list = []
            outputs_pv_seg_list = []

            for i in range(len(hs_list)):
                hs = hs_list[i].permute(0, 2, 1, 3)
                outputs_classes_one2one = []
                outputs_coords_one2one = []
                outputs_pts_coords_one2one = []

                outputs_classes_one2many = []
                outputs_coords_one2many = []
                outputs_pts_coords_one2many = []
                for lvl in range(hs.shape[0]):
                    if lvl == 0:
                        reference = init_reference_list[i][...,0:2] if not self.z_cfg['gt_z_flag'] else init_reference_list[i][...,0:3]
                    else:
                        reference = inter_references_list[i][lvl - 1][...,0:2] if not self.z_cfg['gt_z_flag'] else inter_references_list[i][lvl - 1][...,0:3]
                    reference = inverse_sigmoid(reference)
                    outputs_class = self.cls_branches[lvl](hs[lvl]
                                                    .view(bs, self.num_vec, self.num_pts_per_vec, -1)
                                                    .mean(2))
                    tmp = self.reg_branches[lvl](hs[lvl])
                    tmp = tmp[..., 0:2] if not self.z_cfg['gt_z_flag'] else tmp[..., 0:3]
                    # TODO: check the shape of reference
                    # assert reference.shape[-1] == 2
                    tmp += reference
                    tmp = tmp.sigmoid() # cx,cy,w,h
                    outputs_coord, outputs_pts_coord = self.transform_box(tmp, num_vec=num_vec)
                    outputs_classes_one2one.append(outputs_class[:, 0:self.num_vec_one2one])
                    outputs_coords_one2one.append(outputs_coord[:, 0:self.num_vec_one2one])
                    outputs_pts_coords_one2one.append(outputs_pts_coord[:, 0:self.num_vec_one2one])

                    outputs_classes_one2many.append(outputs_class[:, self.num_vec_one2one:])
                    outputs_coords_one2many.append(outputs_coord[:, self.num_vec_one2one:])
                    outputs_pts_coords_one2many.append(outputs_pts_coord[:, self.num_vec_one2one:])

                outputs_classes_list.append(torch.stack(outputs_classes_one2one))
                outputs_coords_list.append(torch.stack(outputs_coords_one2one))
                outputs_pts_coords_list.append(torch.stack(outputs_pts_coords_one2one))

                outputs_classes_one2many_list.append(torch.stack(outputs_classes_one2many))
                outputs_coords_one2many_list.append(torch.stack(outputs_coords_one2many))
                outputs_pts_coords_one2many_list.append(torch.stack(outputs_pts_coords_one2many))

                outputs_seg = None
                outputs_pv_seg = None
                if self.aux_seg['use_aux_seg']:
                    if self.aux_seg['bev_seg']:
                        outputs_seg = self.seg_head(bev_embed_list[i].float())
                    bs, num_cam, embed_dims, feat_h, feat_w = mlvl_feats[-1].shape
                    if self.aux_seg['pv_seg']:
                        outputs_pv_seg = self.pv_seg_head(mlvl_feats[-1].flatten(0,1))
                        outputs_pv_seg = outputs_pv_seg.view(bs, num_cam, -1, feat_h, feat_w)
                outputs_seg_list.append(outputs_seg)
                outputs_pv_seg_list.append(outputs_pv_seg)
            outs = {
                'bev_embed': bev_embed_list,
                'all_cls_scores': outputs_classes_list,
                'all_bbox_preds': outputs_coords_list,
                'all_pts_preds': outputs_pts_coords_list,
                'dense_heatmap': heatmap_list,
                'depth': depth,
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
                'enc_pts_preds': None,
                'seg': outputs_seg_list,
                'pv_seg': outputs_pv_seg_list,
                "one2many_outs": dict(
                    all_cls_scores=outputs_classes_one2many_list,
                    all_bbox_preds=outputs_coords_one2many_list,
                    all_pts_preds=outputs_pts_coords_one2many_list,
                    dense_heatmap=heatmap_list,
                    enc_cls_scores=None,
                    enc_bbox_preds=None,
                    enc_pts_preds=None,
                    seg=None,
                    pv_seg=None,
                )
            }
            return outs
        else:
            depth_pre = None
            if len(outputs) == 6:
                bev_embed, depth, hs, inter_references, heatmap_list, depth_pre = outputs
            else:
                bev_embed, depth, hs, inter_references, heatmap_list = outputs
            hs = hs.permute(0, 2, 1, 3)
            outputs_classes_one2one = []
            outputs_coords_one2one = []
            outputs_pts_coords_one2one = []

            outputs_classes_one2many = []
            outputs_coords_one2many = []
            outputs_pts_coords_one2many = []
            for lvl in range(hs.shape[0]):
                if lvl == 0:
                    # vec_embedding = hs[lvl].reshape(bs, self.num_vec, -1)
                    outputs_class = self.cls_branches[lvl](hs[lvl]
                                                    .view(bs, num_vec, self.num_pts_per_vec, -1)
                                                    .mean(2))
                    tmp = self.reg_branches[lvl](hs[lvl])
                    # TODO: check the shape of reference
                    #assert reference.shape[-1] == 2
                    tmp = tmp[..., 0:2] if not self.z_cfg['gt_z_flag'] else tmp[..., 0:3]
                    tmp = tmp.sigmoid() # cx,cy,w,h
                else:
                    if self.use_sigmoid:
                        reference = inter_references[lvl - 1][..., 0:2]
                        reference = inverse_sigmoid(reference)
                        outputs_class = self.cls_branches[lvl](hs[lvl]
                                                        .view(bs, num_vec, self.num_pts_per_vec, -1)
                                                        .mean(2))
                        tmp = self.reg_branches[lvl](hs[lvl])

                        # TODO: check the shape of reference
                        #assert reference.shape[-1] == 2
                        tmp = tmp[..., 0:2] if not self.z_cfg['gt_z_flag'] else tmp[..., 0:3]
                        tmp += reference
                        tmp = tmp.sigmoid() # cx,cy,w,h
                    else:
                        reference = inter_references[lvl - 1][..., 0:2]
                        outputs_class = self.cls_branches[lvl](hs[lvl]
                                                        .view(bs, num_vec, self.num_pts_per_vec, -1)
                                                        .mean(2))
                        tmp = self.reg_branches[lvl](hs[lvl])

                        # TODO: check the shape of reference
                        #assert reference.shape[-1] == 2
                        tmp = tmp[..., 0:2] if not self.z_cfg['gt_z_flag'] else tmp[..., 0:3]
                        tmp[..., 0] /= (self.pc_range[3] - self.pc_range[0])
                        tmp[..., 1] /= (self.pc_range[4] - self.pc_range[1])
                        tmp += reference

                outputs_coord, outputs_pts_coord = self.transform_box(tmp, num_vec=num_vec)
                outputs_classes_one2one.append(outputs_class[:, 0:self.num_vec_one2one])
                outputs_coords_one2one.append(outputs_coord[:, 0:self.num_vec_one2one])
                outputs_pts_coords_one2one.append(outputs_pts_coord[:, 0: self.num_vec_one2one])

                outputs_classes_one2many.append(outputs_class[:, self.num_vec_one2one:])
                outputs_coords_one2many.append(outputs_coord[:, self.num_vec_one2one:])
                outputs_pts_coords_one2many.append(outputs_pts_coord[:, self.num_vec_one2one:])

            outputs_classes_one2one = torch.stack(outputs_classes_one2one)
            outputs_coords_one2one = torch.stack(outputs_coords_one2one)
            outputs_pts_coords_one2one = torch.stack(outputs_pts_coords_one2one)

            outputs_classes_one2many = torch.stack(outputs_classes_one2many)
            outputs_coords_one2many = torch.stack(outputs_coords_one2many)
            outputs_pts_coords_one2many = torch.stack(outputs_pts_coords_one2many)

            outputs_seg = None
            outputs_pv_seg = None
            if self.aux_seg['use_aux_seg']:
                if self.aux_seg['bev_seg']:
                    outputs_seg = self.seg_head(bev_embed.float())
                bs, num_cam, embed_dims, feat_h, feat_w = mlvl_feats[-1].shape
                if self.aux_seg['pv_seg']:
                    outputs_pv_seg = self.pv_seg_head(mlvl_feats[-1].flatten(0,1))
                    outputs_pv_seg = outputs_pv_seg.view(bs, num_cam, -1, feat_h, feat_w)
            if depth_pre is not None:
                outs = {
                    'bev_embed': bev_embed,
                    'all_cls_scores': outputs_classes_one2one,
                    'all_bbox_preds': outputs_coords_one2one,
                    'all_pts_preds': outputs_pts_coords_one2one,
                    'dense_heatmap': heatmap_list,
                    'depth': depth,
                    'depth_pre':depth_pre,
                    'enc_cls_scores': None,
                    'enc_bbox_preds': None,
                    'enc_pts_preds': None,
                    'seg': outputs_seg,
                    'pv_seg': outputs_pv_seg,
                    "one2many_outs": dict(
                        all_cls_scores=outputs_classes_one2many,
                        all_bbox_preds=outputs_coords_one2many,
                        all_pts_preds=outputs_pts_coords_one2many,
                        dense_heatmap=heatmap_list,
                        enc_cls_scores=None,
                        enc_bbox_preds=None,
                        enc_pts_preds=None,
                        seg=None,
                        pv_seg=None,
                    )
                }
            else:
                outs = {
                    'bev_embed': bev_embed,
                    'all_cls_scores': outputs_classes_one2one,
                    'all_bbox_preds': outputs_coords_one2one,
                    'all_pts_preds': outputs_pts_coords_one2one,
                    'dense_heatmap': heatmap_list,
                    'depth': depth,
                    'enc_cls_scores': None,
                    'enc_bbox_preds': None,
                    'enc_pts_preds': None,
                    'seg': outputs_seg,
                    'pv_seg': outputs_pv_seg,
                    "one2many_outs": dict(
                        all_cls_scores=outputs_classes_one2many,
                        all_bbox_preds=outputs_coords_one2many,
                        all_pts_preds=outputs_pts_coords_one2many,
                        dense_heatmap=heatmap_list,
                        enc_cls_scores=None,
                        enc_bbox_preds=None,
                        enc_pts_preds=None,
                        seg=None,
                        pv_seg=None,
                    )
            }

            return outs

    def transform_box(self, pts, num_vec=50, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        if self.z_cfg['gt_z_flag']:
            pts_reshape = pts.view(pts.shape[0], num_vec, self.num_pts_per_vec,3)
        else:
            pts_reshape = pts.view(pts.shape[0], num_vec, self.num_pts_per_vec,2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == 'minmax':

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # pts_sampling_result = self.sampler.sample(assign_result, pts_pred,
        #                                       gt_pts)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # pts targets
        # pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
        bbox_weights_list, pts_targets_list, pts_weights_list,
        pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list, pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    pts_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,pts_preds_list,
                                           gt_bboxes_list, gt_labels_list,gt_shifts_pts_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
        pts_targets_list, pts_weights_list, num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        # bbox_weights = bbox_weights * self.code_weights

        # loss_bbox = self.loss_bbox(
        #     bbox_preds[isnotnan, :4], normalized_bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4],
        #     avg_factor=num_total_pos)

        # regression pts CD loss
        # pts_preds = pts_preds

        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range) if not self.z_cfg['gt_z_flag'] \
                                else normalize_3d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2),pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        # loss_diff_pts = self.loss_diff_pts(
        #     pts_preds[isnotnan,:,:], normalized_pts_targets[isnotnan, :,:], cls_scores[isnotnan], labels[isnotnan],
        #     pts_weights[isnotnan,:,:],
        #     avg_factor=num_total_pos)

        if self.seg_line_coord_loss:
            coord_preds = pts_preds[isnotnan,:,:][labels[isnotnan]==1]
            normalized_coord_targets = normalized_pts_targets[isnotnan, :,:][labels[isnotnan]==1]
            coord_weights = pts_weights[isnotnan,:,:][labels[isnotnan]==1]
            num_coord_total_pos = coord_preds.size(0)
            cls_coord_scores = cls_scores[isnotnan][labels[isnotnan]==1]
            coord_labels = labels[isnotnan][labels[isnotnan]==1]

            pts_preds = pts_preds[isnotnan,:,:][labels[isnotnan]!=1]
            normalized_pts_targets = normalized_pts_targets[isnotnan, :,:][labels[isnotnan]!=1]
            pts_weights = pts_weights[isnotnan,:,:][labels[isnotnan]!=1]
            num_pts_total_pos = num_total_pos - coord_preds.size(0)

            loss_pts = self.loss_pts(pts_preds, normalized_pts_targets, pts_weights, avg_factor=num_pts_total_pos)
            loss_coords = self.loss_coords(coord_preds, normalized_coord_targets, coord_weights, avg_factor=num_coord_total_pos)
            loss_coords_diff = self.loss_coords_diff(coord_preds, normalized_coord_targets, cls_coord_scores, coord_labels, coord_weights, avg_factor=num_coord_total_pos)
            if self.use_coords_diff:
                loss_coords += loss_coords_diff
            loss_pts += loss_coords
        else:
            pts_preds = pts_preds[isnotnan,:,:]
            normalized_pts_targets = normalized_pts_targets[isnotnan, :,:]
            pts_weights = pts_weights[isnotnan,:,:]
            loss_pts = self.loss_pts(pts_preds, normalized_pts_targets, pts_weights, avg_factor=num_total_pos)

        # dir_weights = pts_weights[:, :-self.dir_interval,0]
        # denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range) if not self.z_cfg['gt_z_flag'] \
        #                         else denormalize_3d_pts(pts_preds, self.pc_range)
        # denormed_pts_preds_dir = denormed_pts_preds[:,self.dir_interval:,:] - denormed_pts_preds[:,:-self.dir_interval,:]
        # pts_targets_dir = pts_targets[:, self.dir_interval:,:] - pts_targets[:,:-self.dir_interval,:]
        # dir_weights = pts_weights[:, indice,:-1,0]
        # loss_dir = self.loss_dir(
        #     denormed_pts_preds_dir[isnotnan,:,:], pts_targets_dir[isnotnan, :,:],
        #     dir_weights[isnotnan,:],
        #     avg_factor=num_total_pos)

        # bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        # loss_iou = self.loss_iou(
        #     bboxes[isnotnan, :4], bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4],
        #     avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_cls = torch.nan_to_num(loss_cls)
            # loss_bbox = torch.nan_to_num(loss_bbox)
            # loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            #loss_dir = torch.nan_to_num(loss_dir)
            #loss_diff_pts = torch.nan_to_num(loss_diff_pts)
        return loss_cls, loss_pts #, loss_dir

    def _get_heatmap_single(self, gt_shifts_pts, gt_labels):
        gt_heatmap = gt_shifts_pts.new_zeros(self.num_classes, self.bev_h, self.bev_w)
        gt_pts = copy.deepcopy(gt_shifts_pts)
        pc_range = self.train_cfg['point_cloud_range']
        voxel_size = self.train_cfg['voxel_size']
        out_size_factor = ((pc_range[3] - pc_range[0]) / voxel_size[0]) // self.bev_w

        gt_pts[:, 0, :, 0] = ((gt_pts[:, 0, :, 0] - pc_range[0]) / voxel_size[0] / out_size_factor)
        gt_pts[:, 0, :, 1] = ((gt_pts[:, 0, :, 1] - pc_range[1]) / voxel_size[1] / out_size_factor)
        for cls in range(self.num_classes):
            mask_label = (gt_labels == cls)
            gt_tmp = gt_pts[mask_label, 0, :, :].long()
            if gt_tmp.size(0) == 0:
                continue
            if self.coords_version == 'v1':
                for index in range(torch.nonzero(mask_label).size(0)):
                    gt_heatmap[cls] = self.draw_heatmap_line(gt_heatmap[cls], gt_tmp[index], k=1)
            else:
                for index in range(torch.nonzero(mask_label).size(0)):
                    if cls == 1:
                        gt_heatmap[cls] = self.draw_heatmap_line(gt_heatmap[cls], gt_tmp[index], k=1)
                    else:
                        gt_heatmap[cls] = self.draw_heatmap_line(gt_heatmap[cls], gt_tmp[index], k=1)
            for i in range(self.kernel_size):
                gt_heatmap[cls] = self.dilate_heatmap(gt_heatmap[cls], 3)
        #import cv2; cv2.imwrite('gt_heatmap_tmp2.jpg', gt_heatmap.max(dim=0)[0].detach().cpu().numpy()*255)
        return [gt_heatmap]

    def gaussian_blur(self, input_image, kernel_size=5, sigma=1.0):
        """
        高斯模糊函数
        :param input_image: 输入热力图 (H, W)
        :param kernel_size: 高斯核大小
        :param sigma: 高斯分布的标准差
        :return: 高斯模糊后的热力图 (H, W)
        """
        # 确保 kernel_size 是奇数
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # 生成高斯核
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        x = x.to(input_image.device)
        kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()  # 一维归一化
        kernel = kernel[:, None] * kernel[None, :]  # 二维高斯核（自动归一化）

        # 将高斯核转换为卷积核
        kernel = kernel.view(1, 1, kernel_size, kernel_size)

        # 应用卷积操作
        input_image = input_image.unsqueeze(0).unsqueeze(0)  # 增加 batch 和 channel 维度
        output = F.conv2d(input_image, kernel, padding=kernel_size//2)

        return output.squeeze()

    def dilate_heatmap(self, heatmap, kernel_size):
        # 确保heatmap是浮点数类型，以便进行计算
        heatmap = heatmap.float()
        # 应用高斯膨胀
        dilated = self.gaussian_blur(heatmap, kernel_size=kernel_size, sigma=kernel_size / 2)
        # 将原始heatmap的1值位置的膨胀值设置为1，其余位置根据高斯分布计算
        dilated[heatmap == 1] = 1
        dilated = torch.clamp(dilated, 0, 1)
        return dilated

    def rasterize_polygon(self, heatmap, polygon):
        """
        使用射线法并行化判断像素是否在多边形内部
        Args:
            heatmap: torch.Tensor of shape (H, W)
            polygon: torch.Tensor of shape (N, 2) 多边形顶点坐标(x, y)
        Returns:
            mask: torch.Tensor of shape (H, W) 多边形内部为1，外部为0
        """
        h, w = heatmap.shape
        device = heatmap.device

        # 生成网格坐标 (H, W, 2)
        y_grid, x_grid = torch.meshgrid(torch.arange(h, device=device),
                                        torch.arange(w, device=device),
                                        indexing='ij')
        grid_points = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2)  # (H*W, 2)

        # 初始化交叉计数器
        count = torch.zeros(grid_points.shape[0], dtype=torch.int32, device=device)

        n_vertices = polygon.shape[0]
        polygon = polygon.to(device)

        for i in range(n_vertices):
            # 获取当前边两个顶点
            p1 = polygon[i]
            p2 = polygon[(i + 1) % n_vertices]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]

            # 跳过水平边
            delta_y = y2 - y1
            if delta_y == 0:
                continue

            # 并行计算所有点的y坐标比较结果
            with torch.no_grad():  # 不跟踪梯度
                y_compare = grid_points[:, 1]
                mask = (y1 > y_compare) != (y2 > y_compare)

            # 计算交点x坐标（使用广播计算）
            delta_x = x2 - x1
            x_intersect = ( (y_compare - y1) * delta_x / delta_y ) + x1

            # 更新满足条件的交叉计数
            valid_intersect = (x_intersect > grid_points[:, 0]) & mask
            count += valid_intersect.int()

        # 生成最终掩码（奇数次交叉为内部）
        mask = (count % 2 == 1).reshape(h, w).float()
        return mask


    def draw_heatmap_line(self, heatmap, all_points, k):
        mask_max = (all_points[:, 0] == heatmap.size(1))
        all_points[mask_max, 0] -= 1
        mask_max = (all_points[:, 1] == heatmap.size(0))
        all_points[mask_max, 1] -= 1
        heatmap[all_points[:, 1], all_points[:, 0]] = k
        for i in range(all_points.size(0) - 1):
            x1, y1 = all_points[i, :]
            x2, y2 = all_points[i+1, :]
            x, y = x1, y1
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            sx = -1 if x1 > x2 else 1
            sy = -1 if y1 > y2 else 1
            if dx > dy:
                err = dx / 2.0
                while x != x2:
                    heatmap[y, x] = k
                    err -= dy
                    if err < 0:
                        y += sy
                        err += dx
                    x += sx
            else:
                err = dy / 2.0
                while y != y2:
                    heatmap[y, x] = k
                    err -= dx
                    if err < 0:
                        x += sx
                        err += dy
                    y += sy
            heatmap[y, x] = k
        return heatmap

    def gener_heatmap_gt(self, dense_heatmap, gt_shifts_pts_list, gt_labels_list):
        gt_heatmap_batch = multi_apply(self._get_heatmap_single, gt_shifts_pts_list, gt_labels_list)
        gt_heatmap_batch = torch.stack(gt_heatmap_batch[0])
        # import cv2
        # #cv2.imwrite('pred.jpg', clip_sigmoid(dense_heatmap)[0].sum(dim=0).detach().cpu().numpy() * 255)
        # cv2.imwrite('gt.jpg', gt_heatmap_batch[0].sum(dim=0).detach().cpu().numpy() * 255)
        loss_heatmap = self.loss_heatmap(clip_sigmoid(dense_heatmap), gt_heatmap_batch, avg_factor=max(gt_heatmap_batch.eq(1).float().sum().item(), 1))
        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            loss_heatmap = torch.nan_to_num(loss_heatmap)
        return loss_heatmap

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             gt_seg_mask,
             gt_pv_seg_mask,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        gt_vecs_list = copy.deepcopy(gt_bboxes_list)
        if self.use_fast_perception_neck:
            head_weights = [0.5, 0.5, 1.0]
            loss_dict = dict()
            for head_id in range(len(preds_dicts['all_cls_scores'])):
                all_cls_scores = preds_dicts['all_cls_scores'][head_id]
                all_bbox_preds = preds_dicts['all_bbox_preds'][head_id]
                all_pts_preds  = preds_dicts['all_pts_preds'][head_id]
                enc_cls_scores = None
                enc_bbox_preds = None
                enc_pts_preds  = None

                num_dec_layers = len(all_cls_scores)  # 6
                device = gt_labels_list[0].device

                gt_bboxes_list = [
                    gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]
                gt_pts_list = [
                    gt_bboxes.fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
                if self.gt_shift_pts_pattern == 'v0':
                    gt_shifts_pts_list = [
                        gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
                elif self.gt_shift_pts_pattern == 'v1':
                    gt_shifts_pts_list = [
                        gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in gt_vecs_list]
                elif self.gt_shift_pts_pattern == 'v2':
                    gt_shifts_pts_list = [
                        gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in gt_vecs_list]
                elif self.gt_shift_pts_pattern == 'v3':
                    gt_shifts_pts_list = [
                        gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in gt_vecs_list]
                elif self.gt_shift_pts_pattern == 'v4':
                    gt_shifts_pts_list = [
                        gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in gt_vecs_list]
                else:
                    raise NotImplementedError

                all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
                all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
                all_gt_pts_list = [gt_pts_list for _ in range(num_dec_layers)]
                all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
                all_gt_bboxes_ignore_list = [
                    gt_bboxes_ignore for _ in range(num_dec_layers)
                ]
                losses_cls, losses_pts, loss_coords, losses_dir, losses_diff_pts = multi_apply(
                    self.loss_single, all_cls_scores, all_bbox_preds, all_pts_preds,
                    all_gt_bboxes_list, all_gt_labels_list, all_gt_shifts_pts_list,
                    all_gt_bboxes_ignore_list)
                if self.aux_seg['use_aux_seg']:
                    if self.aux_seg['bev_seg']:
                        if preds_dicts['seg'] is not None:
                            seg_output = preds_dicts['seg'][head_id]
                            num_imgs = seg_output.size(0)
                            seg_gt = torch.stack([gt_seg_mask[i] for i in range(num_imgs)],dim=0)
                            loss_seg = self.loss_seg(seg_output, seg_gt.float())
                            loss_dict[f'{head_id}_loss_seg'] = loss_seg
                    if self.aux_seg['pv_seg']:
                        if preds_dicts['pv_seg'] is not None:
                            pv_seg_output = preds_dicts['pv_seg'][head_id]
                            num_imgs = pv_seg_output.size(0)
                            pv_seg_gt = torch.stack([gt_pv_seg_mask[i] for i in range(num_imgs)],dim=0)
                            loss_pv_seg = self.loss_pv_seg(pv_seg_output, pv_seg_gt.float())
                            loss_dict[f'{head_id}_loss_pv_seg'] = loss_pv_seg

                # loss from the last decoder layer
                loss_dict[f'{head_id}_loss_cls'] = losses_cls[-1] * head_weights[head_id]
                loss_dict[f'{head_id}_loss_pts'] = losses_pts[-1] * head_weights[head_id]
                loss_dict[f'{head_id}_loss_dir'] = losses_dir[-1] * head_weights[head_id]
                loss_dict[f'{head_id}_losses_diff_pts'] = losses_diff_pts[-1] * head_weights[head_id]
                # loss from other decoder layers
                num_dec_layer = 0
                for loss_cls_i, loss_bbox_i, loss_iou_i, loss_pts_i, loss_dir_i, loss_diff_i in zip(losses_cls[:-1],
                                                losses_pts[:-1],
                                                losses_dir[:-1],
                                                losses_diff_pts[:-1]):
                    loss_dict[f'{head_id}_d{num_dec_layer}.loss_cls'] = loss_cls_i * head_weights[head_id]
                    loss_dict[f'{head_id}_d{num_dec_layer}.loss_pts'] = loss_pts_i * head_weights[head_id]
                    loss_dict[f'{head_id}_d{num_dec_layer}.loss_dir'] = loss_dir_i * head_weights[head_id]
                    loss_dict[f'{head_id}_d{num_dec_layer}._losses_diff_pts'] = loss_diff_i * head_weights[head_id]
                    num_dec_layer += 1
        else:
            all_cls_scores = preds_dicts['all_cls_scores']
            all_bbox_preds = preds_dicts['all_bbox_preds']
            all_pts_preds  = preds_dicts['all_pts_preds']
            if 'dense_heatmap' in preds_dicts:
                dense_heatmaps = preds_dicts['dense_heatmap']
            enc_cls_scores = preds_dicts['enc_cls_scores']
            enc_bbox_preds = preds_dicts['enc_bbox_preds']
            enc_pts_preds  = preds_dicts['enc_pts_preds']

            num_dec_layers = len(all_cls_scores)
            device = gt_labels_list[0].device

            gt_bboxes_list = [
                gt_bboxes.bbox.to(device) for gt_bboxes in gt_vecs_list]
            gt_pts_list = [
                gt_bboxes.fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
            if self.gt_shift_pts_pattern == 'v0':
                gt_shifts_pts_list = [
                    gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in gt_vecs_list]
            elif self.gt_shift_pts_pattern == 'v1':
                gt_shifts_pts_list = [
                    gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in gt_vecs_list]
            elif self.gt_shift_pts_pattern == 'v2':
                gt_shifts_pts_list = [
                    gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in gt_vecs_list]
            elif self.gt_shift_pts_pattern == 'v3':
                gt_shifts_pts_list = [
                    gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in gt_vecs_list]
            elif self.gt_shift_pts_pattern == 'v4':
                gt_shifts_pts_list = [
                    gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in gt_vecs_list]
            else:
                raise NotImplementedError
            all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
            all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
            all_gt_pts_list = [gt_pts_list for _ in range(num_dec_layers)]
            all_gt_shifts_pts_list = [gt_shifts_pts_list for _ in range(num_dec_layers)]
            all_gt_bboxes_ignore_list = [
                gt_bboxes_ignore for _ in range(num_dec_layers)
            ]
            if 'dense_heatmap' in preds_dicts:
                loss_heatmap = self.gener_heatmap_gt(dense_heatmaps, gt_shifts_pts_list, gt_labels_list)

            losses_cls, losses_pts = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,all_pts_preds,
                all_gt_bboxes_list, all_gt_labels_list,all_gt_shifts_pts_list,
                all_gt_bboxes_ignore_list)

            loss_dict = dict()
            # loss of proposal generated from encode feature map.
            if self.aux_seg['use_aux_seg']:
                if self.aux_seg['bev_seg']:
                    if preds_dicts['seg'] is not None:
                        seg_output = preds_dicts['seg']
                        num_imgs = seg_output.size(0)
                        seg_gt = torch.stack([gt_seg_mask[i] for i in range(num_imgs)],dim=0)
                        loss_seg = self.loss_seg(seg_output, seg_gt.float())
                        loss_dict['loss_seg'] = loss_seg
                if self.aux_seg['pv_seg']:
                    if preds_dicts['pv_seg'] is not None:
                        pv_seg_output = preds_dicts['pv_seg']
                        num_imgs = pv_seg_output.size(0)
                        pv_seg_gt = torch.stack([gt_pv_seg_mask[i] for i in range(num_imgs)],dim=0)
                        loss_pv_seg = self.loss_pv_seg(pv_seg_output, pv_seg_gt.float())
                        loss_dict['loss_pv_seg'] = loss_pv_seg
            if self.use_start_pts:
                # loss from the last decoder layer
                if self.epoch >= self.heatmap_loss_stop_epoch:
                    loss_dict['loss_cls'] = losses_cls[-1] * self.multi_loss_weight[-1]
                    loss_dict['loss_pts'] = losses_pts[-1] * self.multi_loss_weight[-1]
                if 'dense_heatmap' in preds_dicts:
                    loss_dict['loss_heatmap'] = loss_heatmap
                #loss_dict['loss_dir'] = losses_dir[-1]
                # loss from other decoder layers
                num_dec_layer = 0
                for index,(loss_cls_i, loss_pts_i) in enumerate(zip(losses_cls[:-1],
                                                losses_pts[:-1])):
                    if self.epoch >= self.heatmap_loss_stop_epoch:
                        loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i * self.multi_loss_weight[index]
                        loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i * self.multi_loss_weight[index]
                    loss_dict[f'd{num_dec_layer}.loss_heatmap'] = loss_heatmap
                    #loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
                    num_dec_layer += 1
            else:
                # loss from the last decoder layer
                loss_dict['loss_cls'] = losses_cls[-1] * self.multi_loss_weight[-1]
                loss_dict['loss_pts'] = losses_pts[-1] * self.multi_loss_weight[-1]
                if 'dense_heatmap' in preds_dicts and self.epoch < self.heatmap_loss_stop_epoch:
                    loss_dict['loss_heatmap'] = loss_heatmap
                #loss_dict['loss_dir'] = losses_dir[-1]
                # loss from other decoder layers
                num_dec_layer = 0
                for index,(loss_cls_i, loss_pts_i) in enumerate(zip(losses_cls[:-1],
                                                losses_pts[:-1])):
                    loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i * self.multi_loss_weight[index]
                    loss_dict[f'd{num_dec_layer}.loss_pts'] = loss_pts_i * self.multi_loss_weight[index]
                    if self.epoch < self.heatmap_loss_stop_epoch:
                        loss_dict[f'd{num_dec_layer}.loss_heatmap'] = loss_heatmap
                    #loss_dict[f'd{num_dec_layer}.loss_dir'] = loss_dir_i
                    num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # bboxes: xmin, ymin, xmax, ymax
        if self.use_fast_perception_neck:
            preds_dicts['all_cls_scores'] = preds_dicts['all_cls_scores'][-1]
            preds_dicts['all_bbox_preds'] = preds_dicts['all_bbox_preds'][-1]
            preds_dicts['all_pts_preds']  = preds_dicts['all_pts_preds'][-1]
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            scores = preds['scores']
            labels = preds['labels']
            pts = preds['pts']

            ret_list.append([bboxes, scores, labels, pts])

        return ret_list

