# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
# For nuScenes we usually do 10-class detection
class_names = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
               'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
vector_classes = ['divider', 'ped_crossing','boundary']
occupancy_classes = ['void', 'static', 'motional']

dataset_type = 'MYNuScenesDataset'
data_root = '/dahuafs/groupdata/share/openset/nuscenes/'
seed=0
deterministic=False
# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

reduce_beams=32
load_dim=5
use_dim=5
bev_h_ = 360
bev_w_ = 360
queue_length = 1

fixed_ptsnum_per_gt_line = 20  # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag=True

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='CustomLoadPointsFromFile', coord_type='LIDAR', load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams),
    dict(type='CustomLoadPointsFromMultiSweeps', sweeps_num=9, load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams, pad_empty_sweeps=True, remove_close=True),
    dict(type='CustomPointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(
        type='LoadOccupancyAnnotations3D',
        occupancy_path='/dahuafs/groupdata/Lidaralgorithm/nuscenes_occupancy_labels/occupancy_heteromorphic/semantic_3class_train/',
        occupancy_label_range=point_cloud_range,
        occupancy_label_size=[0.15, 0.15, 0.2],
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='BEVGlobalRotScaleTrans', resize_lim=[1.0, 1.0], rot_lim=[0.0, 0.0], trans_lim=0.0, is_train=True),
    #dict(type='BEVRandomFlip3D'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'points', 'occupancy_labels'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(
        type='LoadOccupancyAnnotations3D',
        occupancy_path='/dahuafs/groupdata/Lidaralgorithm/nuscenes_occupancy_labels/occupancy_heteromorphic/semantic_3class_val/',
        occupancy_label_range=point_cloud_range,
        occupancy_label_size=[0.15, 0.15, 0.2],
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='BEVGlobalRotScaleTrans', resize_lim=[1.0, 1.0], rot_lim=[0.0, 0.0], trans_lim=0.0, is_train=True),
    dict(type='CustomLoadPointsFromFile', coord_type='LIDAR', load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams),
    dict(type='CustomLoadPointsFromMultiSweeps', sweeps_num=9, load_dim=load_dim, use_dim=use_dim, reduce_beams=reduce_beams, pad_empty_sweeps=True, remove_close=True),
    dict(type='CustomPointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img', 'points', 'occupancy_labels'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
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
        vector_classes=vector_classes,
        occupancy_classes=occupancy_classes,
        queue_length=queue_length,
        box_type_3d='LiDAR',
        epoch_cur=0),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '20704_nuscenes_detection_vector_infos_val.pkl',
        map_ann_file=data_root + 'nuscenes_map_anns_val.json',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        vector_classes=vector_classes,
        occupancy_classes=occupancy_classes,
        classes=class_names,
        modality=input_modality,
        samples_per_gpu=1),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '20704_nuscenes_detection_vector_infos_val.pkl',
        map_ann_file=data_root + 'nuscenes_map_anns_val.json',
        pipeline=test_pipeline,
        bev_size=(bev_h_, bev_w_),
        pc_range=point_cloud_range,
        fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        padding_value=-10000,
        vector_classes=vector_classes,
        occupancy_classes=occupancy_classes,
        classes=class_names,
        modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

evaluation=dict(interval=1, pipeline=test_pipeline)