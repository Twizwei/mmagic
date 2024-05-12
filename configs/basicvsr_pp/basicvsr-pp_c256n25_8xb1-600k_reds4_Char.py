_base_ = '../basicvsr/basicvsr_2xb4_reds4.py'

experiment_name = 'basicvsr-pp_c64n7_8xb1-600k_reds4_CharLPIPS'
work_dir = f'./work_dirs/{experiment_name}'
save_dir = './work_dirs'

train_pipeline = [
    dict(type='GenerateSegmentIndices', interval_list=[1]),
    dict(type='LoadImageFromFile', key='img', channel_order='rgb'),
    dict(type='LoadImageFromFile', key='gt', channel_order='rgb'),
    dict(type='SetValues', dictionary=dict(scale=4)),
    dict(type='PairedRandomCrop', gt_patch_size=256),
    dict(
        type='Flip',
        keys=['img', 'gt'],
        flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['img', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['img', 'gt'], transpose_ratio=0.5),
    dict(type='MirrorSequence', keys=['img', 'gt']),
    dict(type='PackInputs')
]


# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='BasicVSRPlusPlusNet',
        mid_channels=256,
        num_blocks=25,
        scale_factor=4,
        is_low_res_input=True,
        spynet_pretrained='https://download.openmmlab.com/mmediting/restorers/'
        'basicvsr/spynet_20210409-c6c1bd09.pth'),
    pixel_loss=dict(type='CharbonnierLoss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=0.0,
    train_cfg=dict(fix_iter=5000),
    data_preprocessor=dict(
        type='DataPreprocessor',
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
    ))

train_dataloader = dict(
    num_workers=6, batch_size=1, dataset=dict(num_input_frames=10, pipeline=train_pipeline))

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=600_000, val_interval=5000)

# optimizer
optim_wrapper = dict(
    constructor='DefaultOptimWrapperConstructor',
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.99)),
    paramwise_cfg=dict(custom_keys={'spynet': dict(lr_mult=0.25)}), 
    # bypass_duplicate=True
    )

default_hooks = dict(checkpoint=dict(out_dir=save_dir))

# learning policy
param_scheduler = dict(
    type='CosineRestartLR',
    by_epoch=False,
    periods=[600000],
    restart_weights=[1],
    eta_min=1e-7)
