_base_ = [
    './datasets/custom_nus-3d.py',
    './_base_/default_runtime.py'
]

plugin = True
#plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
point_cloud_range = [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]
lidar_point_cloud_range = [-15.0, -30.0, -5.0, 15.0, 30.0, 3.0]
voxel_size = [0.15, 0.15, 0.2]
dbound=[1.0, 35.0, 0.5]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
# map has classes: divider, ped_crossing, boundary
map_classes = ['divider', 'ped_crossing', 'boundary']
# fixed_ptsnum_per_line = 20
# map_classes = ['divider',]
fixed_ptsnum_per_gt_line = 20 # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 20
num_vec_one2one = 50
num_vec_one2many = 300
eval_use_same_gt_sample_num_flag = True
num_map_classes = len(map_classes)
use_fast_perception_neck = False

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=True,
    seg_classes=1,
    feat_down_sample=32,
    pv_thickness=1,
)

grid_config = {
    'x': [-30.0, -30.0, 0.15], # useless
    'y': [-15.0, -15.0, 0.15], # useless
    'z': [-10, 10, 20],        # useless
    'depth': [1.0, 35.0, 0.5], # useful
}

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
# bev_h_ = 50
# bev_w_ = 50
bev_h_ = 200
bev_w_ = 100
queue_length = 1 # each sequence contains `queue_length` frames.

model = dict(
    type='MapTR',
    use_grid_mask=True,
    video_test_mode=False,
    pretrained=dict(img='ckpts/resnet50-19c8e357.pth'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch'),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True),
    # img_neck=dict(
    #     type='NEWFPN',
    #     in_channels=[512, 1024, 2048],
    #     out_channels=[256, 256, 256],
    #     upsample_strides=[1/2, 1, 2],
    #     norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
    #     upsample_cfg=dict(type='deconv', bias=False),
    #     use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='MapTRHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_vec_one2one=num_vec_one2one,
        num_vec_one2many=num_vec_one2many,
        k_one2many=6,
        multi_loss_weight=[0.5, 1.0],  # two-stage loss weight
        num_pts_per_vec=fixed_ptsnum_per_pred_line, # one bbox
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        kernel_size=30,  #高斯膨胀核
        seg_line_coord_loss=False,  # 分离线和人行横道loss
        coords_version='v1',
        use_coords_diff=False,  # 使用diff loss
        use_sigmoid=True,  # 是否使用sigmod相加各层结果
        heatmap_loss_stop_epoch=12,  # 什么epoch开始停止heatmap loss
        use_start_pts=True,
        multi_loss=True,  # two-stage loss
        query_embed_type='instance_pts',
        transform_method='minmax',
        gt_shift_pts_pattern='v2',
        num_classes=num_map_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        aux_seg=aux_seg_cfg,
        transformer=dict(
            type='MapTRPerceptionHeatmapTransformer',
            bev_h=bev_h_,
            bev_w=bev_w_,
            rotate_prev_bev=True,
            use_fast_perception_neck=use_fast_perception_neck,
            use_shift=True,
            use_can_bus=True,
            heatmap_version='v3',
            heatmap_weight=8.0,
            num_proposals=3500,
            use_add=True,
            use_attention=False,
            attention_version='v1',
            embed_dims=_dim_,
            encoder=dict(
                type='LSSTransformV2',
                input_size=[480, 800],
                in_channels=_dim_,
                out_channels=_dim_,
                feat_down_sample=32,
                pc_range=point_cloud_range,
                voxel_size=[0.15, 0.15, 20],
                dbound=dbound,
                downsample=2,
                depthnet_cfg=dict(use_dcn=False, with_cp=False, aspp_mid_channels=96),
                grid_config=grid_config,
                loss_depth_weight=0.5,
                sid=True,
            ),
            decoder=dict(
                type='MapTRDecoder',
                num_layers=1,
                return_intermediate=True,
                use_res=True,
                transformerlayers=dict(
                    type='DecoupledDetrTransformerDecoderLayer',
                    num_vec=num_vec_one2one,
                    num_pts_per_vec=fixed_ptsnum_per_pred_line,
                    attn_cfgs=[
                        # dict(
                        #     type='MultiheadAttention',
                        #     embed_dims=_dim_,
                        #     num_heads=8,
                        #     dropout=0.1),
                        # dict(
                        #     type='MultiheadAttention',
                        #     embed_dims=_dim_,
                        #     num_heads=8,
                        #     dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=1),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    #operation_order=('self_attn', 'norm', 'self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')))),
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm')))),
        backbone=dict(
            type='SECOND',
            in_channels=256,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
            conv_cfg=dict(type='Conv2d', bias=False)),
        neck=dict(
            type='SECONDFPN',
            in_channels=[128, 256],
            out_channels=[128, 128],
            upsample_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
            upsample_cfg=dict(type='deconv', bias=False),
            use_conv_for_no_stride=True),
        bbox_coder=dict(
            type='MapTRNMSFreeCoder',
            # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=point_cloud_range,
            max_num=50,
            voxel_size=voxel_size,
            num_classes=num_map_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_heatmap = dict(type='WeightGaussianFocalLoss', reduction='mean', loss_weight=1.2, heatmap_weight=0.8, class_weight=[1.0, 1.0, 1.0]),
        loss_pts=dict(type='PtsLineLoss', loss_weight=2.5, func_weight=[1, 1, 1]),
        #loss_coords=dict(type='PtsLineLoss', loss_weight=0.15, func_weight=[0, 1, 1]),
        loss_coords=dict(type='PtsL1Loss', loss_weight=0.15),
        loss_coords_diff=dict(type='DiffPtsL1Loss', loss_weight=0.05),
        loss_seg=dict(type='SimpleLoss',
            pos_weight=4.0,
            loss_weight=1.0),
        loss_pv_seg=dict(type='SimpleLoss',
                    pos_weight=1.0,
                    loss_weight=2.0),
        ),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='MapTRAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsLineCost', weight=5.0, func_weight=[1, 1, 1]),
            pc_range=point_cloud_range))))

dataset_type = 'MYNuScenesDataset'
data_root = '/dahuafs/groupdata/share/openset/nuscenes/'
file_client_args = dict(backend='disk')

reduce_beams=32
load_dim=5
use_dim=5

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CustomLoadPointsFromFile', coord_type='LIDAR', load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams),
    dict(type='CustomLoadPointsFromMultiSweeps', sweeps_num=9, load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams, pad_empty_sweeps=True, remove_close=True),
    dict(type='CustomPointsRangeFilter', point_cloud_range=lidar_point_cloud_range),
    #dict(type='GlobalRotScaleTransMap', resize_lim=[0.9, 1.1], rot_lim=[-0.78539816, 0.78539816], trans_lim=0.5, is_train=True),
    dict(type='GlobalRotScaleTransMap', resize_lim=[1.0, 1.0], rot_lim=[-0.0, 0.0], trans_lim=0.0, stop_epoch=15, is_train=True),
    #dict(type='RandomFlip3DMap'),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='CustomPointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='PadMultiViewImageDepth', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['img', 'points', 'gt_depth'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='CustomLoadPointsFromFile', coord_type='LIDAR', load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams),
    dict(type='CustomLoadPointsFromMultiSweeps', sweeps_num=9, load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams, pad_empty_sweeps=True, remove_close=True),
    dict(type='GlobalRotScaleTransMap', resize_lim=[1.0, 1.0], rot_lim=[-0.0, 0.0], trans_lim=0.0, is_train=False),
    dict(type='CustomPointsRangeFilter', point_cloud_range=lidar_point_cloud_range),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img', 'points'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '20704_nuscenes_detection_vector_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        vector_classes=map_classes,
        queue_length=queue_length,
        aux_seg=aux_seg_cfg,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + '20704_nuscenes_detection_vector_infos_val.pkl',
             map_ann_file=data_root + 'nuscenes_map_anns_val.json',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             pc_range=point_cloud_range,
             fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
             eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
             padding_value=-10000,
             vector_classes=map_classes,
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + '20704_nuscenes_detection_vector_infos_val.pkl',
              map_ann_file=data_root + 'nuscenes_map_anns_val.json',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              pc_range=point_cloud_range,
              fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
              eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
              padding_value=-10000,
              vector_classes=map_classes,
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=1e-3,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 12
# total_epochs = 50
# evaluation = dict(interval=1, pipeline=test_pipeline)
evaluation = dict(interval=12, pipeline=test_pipeline, metric='chamfer')
runner = dict(type='Custom_EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
fp16 = dict(loss_scale=512.)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
custom_hooks = [dict(type='CustomSetEpochInfoHook')]

find_unused_parameters = True
work_dir = 'work_dirs/FastMap_kernel30_lr1e-3_ep12'
load_from = None # 'work_dirs/MapTR_camera_pts2line_guassheatmap_weightheatmap_noresself_3cross/latest.pth'
#resume_from='work_dirs/FastMap_base_decoder1_backboneneck_numpro2500_nosigmoid_rotaugstop20/latest.pth'