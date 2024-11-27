import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    arguments = {
        'dataset': [
            dict(name='--datasetName',
                 type=str,
                 default='sims',
                 help='mosi, mosei, sims or simsv2'),
            dict(name='--dataPath',
                 default="/opt/data/private/Project/Datasets/MSA_Datasets/SIMS/Processed/unaligned_39.pkl",
                 type=str,
                 help=' '),
            dict(name='--seq_lens',
                 default=[39, 55, 400],
                 type=list,
                 help='the length of T, V, A modalities; sims: [39, 55, 400]; simsv2: []; mosi: []; mosei: []'),
            dict(name='--num_workers',
                 default=8,
                 type=int,
                 help=' '),
            dict(name='--train_mode',
                 default="regression",
                 type=str,
                 help=' '),
        ],

        'network': [
            dict(name='--fusion_layers',
                 default=3,
                 type=int),
            dict(name='--dropout',
                 default=0.3,
                 type=float),
            dict(name='--hidden_size',
                 default=256,
                 type=int),
            dict(name='--ffn_size',
                 default=512,
                 type=int)
        ],

        'common': [
            dict(name='--seed',  # try different seeds
                 default=1111,
                 type=int),
            dict(name='--batch_size',
                 default=32,
                 type=int,
                 help=' '),
            dict(name='--lr',
                 type=float,
                 default=3e-5),
            dict(name='--weight_decay',
                 type=float,
                 default=1e-5),
            dict(name='--n_epochs',
                 default=50,
                 type=int,
                 help='Number of total epochs to run'),
            dict(name='--log_path',
                 default='./log/',
                 type=str,
                 help='the logger path for save options and experience results')
        ]
    }

    for group in arguments.values():
        for argument in group:
            name = argument['name']
            del argument['name']
            parser.add_argument(name, **argument)

    args = parser.parse_args()
    return args
