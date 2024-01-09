import argparse
from pathlib import Path

import torch

from models.swiftformer import SwiftFormerEncoder


def main(args):
    print('Setting up SwiftFormerEncoder Block...')
    if args.depth > 1:
        raise NotImplementedError('TODO')
    else:
        net = SwiftFormerEncoder(
            args.dim, mlp_ratio=args.mlp_ratio,
            # act_layer=act_layer, drop_path=block_dpr,
            # use_layer_scale=use_layer_scale,
            # layer_scale_init_value=layer_scale_init_value
        )

    input = torch.rand((args.b_size, args.dim) + args.input_size)
    out = net(input)

    print('Exporting...')
    if not args.output.parent.exists():
        print(f'Making folder: {args.output.parent}')
        args.output.parent.mkdir(exist_ok=True, parents=True)
    torch_out = torch.onnx.export(
        net.cpu(),
        (input),
        args.output,
        verbose=False,
        export_params=True,
        input_names=['input'], output_names=['args.output'],
        do_constant_folding=True,
        opset_version=args.opset_version
        # training=torch.onnx.TrainingMode.EVAL,
    )
    if args.output.exists():
        print(f'🎉 Model exported: {args.output.absolute()}')
    else:
        raise FileNotFoundError(
            '😱 You shall not pass: EXPORT FAILED ⚠️ Good luck 🧑‍💻🔍🐛, wish '
            'you the on-device Gods to be with you!'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Export SwiftFormerEncoder (Attention) Block'
    )
    parser.add_argument('--output', type=Path,
                        default=Path('checkpoints/model.onnx'))
    parser.add_argument('--input_size', type=int, nargs='+', default=(64, 64))
    parser.add_argument('--b_size', type=int, default=1, help='Batch size')
    parser.add_argument('--output_file', action='store_true', default=True,
                        help='Path is a file')
    parser.add_argument('--overwrite', '-f', action='store_true', default=False,
                        help='Overwrite existing onnx')
    parser.add_argument('--opset_version', type=int, default=11,
                        help='ONNX Opset version')
    parser.add_argument('--dim', type=int, default=96, help='Embedding dimension')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio')
    parser.add_argument('--num_heads', type=int, default=1, help='Num heads')
    parser.add_argument('--depth', type=int, default=1, help='Depth parameter')

    args = parser.parse_args()

    # Sanitization & verbosity
    if not isinstance(args.input_size, tuple):
        args.input_size = tuple(args.input_size)
    if args.num_heads != 1:
        raise NotImplementedError('SwiftFormer uses num_heads=1')
        head_dim = args.dim // args.num_heads
        print(f'{head_dim=} := {args.dim=}, {args.num_heads=}')

    # Mobile friendly choices reminder
    # awful_choice = (args.num_heads % 8 != 0 or args.embed_dim % 32 != 0 or
    #                 args.head_dim % 32 != 0)
    # if args.b_size == 1 and awful_choice:
    #     print('❗Awful hyperparemeter setup. Stop assuming GPUs, Samsung biz'
    #         ' is not there❗\n Develop mobile-friendly models'
    #         f'\n{args.num_heads=}, {args.embed_dim=} {args.head_dim=}'
    #     )

    if args.output_file and args.output.suffix != '.onnx':
        args.output = args.output / 'model.onnx'
    if args.output.exists() and not args.overwrite:
        raise ValueError(f'{args.output=} existed. Set --overwrite / -f')

    main(args)
