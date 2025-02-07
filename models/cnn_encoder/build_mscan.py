from .mscan import MSCAN


def build_mscan_tiny(checkpoint=None):
    embed_dims = [32, 64, 160, 256]
    depths = [3, 3, 5, 2]
    drop_path_rate = 0.1

    if checkpoint is not None:
        init_cfg = dict(type="Pretrained", checkpoint=checkpoint)
    else:
        init_cfg = None

    model = MSCAN(
        embed_dims=embed_dims,
        depths=depths,
        drop_path_rate=drop_path_rate,
        init_cfg=init_cfg,
    )
    model.init_weights()

    return model


def build_mscan_small(checkpoint=None):
    embed_dims = [64, 128, 320, 512]
    depths = [2, 2, 4, 2]
    drop_path_rate = 0.1

    if checkpoint is not None:
        init_cfg = dict(type="Pretrained", checkpoint=checkpoint)
    else:
        init_cfg = None

    model = MSCAN(
        embed_dims=embed_dims,
        depths=depths,
        drop_path_rate=drop_path_rate,
        init_cfg=init_cfg,
    )
    model.init_weights()

    return model


def build_mscan_base(checkpoint=None):
    embed_dims = [64, 128, 320, 512]
    depths = [3, 3, 12, 3]
    drop_path_rate = 0.1

    if checkpoint is not None:
        init_cfg = dict(type="Pretrained", checkpoint=checkpoint)
    else:
        init_cfg = None

    model = MSCAN(
        embed_dims=embed_dims,
        depths=depths,
        drop_path_rate=drop_path_rate,
        init_cfg=init_cfg,
    )
    model.init_weights()

    return model


def build_mscan_large(checkpoint=None):
    embed_dims = [64, 128, 320, 512]
    depths = [3, 5, 27, 3]
    drop_path_rate = 0.3

    if checkpoint is not None:
        init_cfg = dict(type="Pretrained", checkpoint=checkpoint)
    else:
        init_cfg = None

    model = MSCAN(
        embed_dims=embed_dims,
        depths=depths,
        drop_path_rate=drop_path_rate,
        init_cfg=init_cfg,
    )
    model.init_weights()

    return model
