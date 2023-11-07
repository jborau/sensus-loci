# model settings
model = dict(
    type='CenterNetMono3D',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='DLANet',
        depth=34,
        in_channels=3,
        norm_cfg=dict(type='GN', num_groups=32),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth'
        )),
    neck=dict(
        type='DLANeck',
        in_channels=[16, 32, 64, 128, 256, 512],
        start_level=2,
        end_level=5,
        norm_cfg=dict(type='GN', num_groups=32)),
    bbox_head=dict(
        type='MonoConHead',
        in_channels=64,
        feat_channels=64,
        num_classes=3,
        num_alpha_bins=12,
        loss_center_heatmap=dict(type='CenterNetGaussianFocalLoss', loss_weight=1.0),
        loss_wh=dict(type='mmdet.L1Loss', loss_weight=0.1),
        loss_offset=dict(type='mmdet.L1Loss', loss_weight=1.0),
        loss_center2kpt_offset=dict(type='mmdet.L1Loss', loss_weight=1.0),
        loss_kpt_heatmap=dict(type='CenterNetGaussianFocalLoss', loss_weight=1.0),
        loss_kpt_heatmap_offset=dict(type='mmdet.L1Loss', loss_weight=1.0),
        loss_dim=dict(type='DimAwareL1Loss', loss_weight=1.0),
        loss_depth=dict(type='LaplacianAleatoricUncertaintyLoss', loss_weight=1.0),
        loss_alpha_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_alpha_reg=dict(type='mmdet.L1Loss', loss_weight=1.0),
    ),
    train_cfg=None,
    test_cfg=dict(topk=30, local_maximum_kernel=3, max_per_img=30, thresh=0.4))
