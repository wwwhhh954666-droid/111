_base_ = [
    r'D:\mask\mmsegmentation-main\configs\_base_\models\segformer_mit-b0.py',
    r'D:\mask\mmsegmentation-main\configs\_base_\default_runtime.py',
]

crop_size = (512, 512)
num_classes = 3
metainfo = dict(
    classes=('background', 'crack', 'spall'),
    palette=[[0, 0, 0], [255, 40, 40], [40, 180, 255]],
)

data_preprocessor = dict(size=crop_size)
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(init_cfg=None),
    decode_head=dict(num_classes=num_classes, norm_cfg=norm_cfg),
)

dataset_type = 'BaseSegDataset'
data_root = r'D:\mask\baseline_runs\patch_dataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, by_epoch=False, begin=0, end=20),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=20, end=600, by_epoch=False),
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=600, val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=100, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)

work_dir = r'D:\mask\baseline_runs\experiments\mmseg_segformer_b0'
randomness = dict(seed=20260605, deterministic=False)
