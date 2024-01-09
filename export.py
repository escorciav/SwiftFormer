import argparse
from importlib import import_module
from pathlib import Path

import torch

from models import swiftformer


def main(args):
    print('Setting up SwiftFormer...')
    net_loader = getattr(swiftformer, args.arch)
    net = net_loader()

    input = torch.rand((args.b_size, args.dim) + args.input_size)
    out = net(input)

    if args.verbose:
        print(net)

    print('Exporting...')
    if not args.output.parent.exists():
        print(f'Making folder: {args.output.parent}')
        args.output.parent.mkdir(exist_ok=True, parents=True)
    torch_out = torch.onnx.export(
        net.cpu(),
        (input),
        args.output,
        verbose=args.verbose,
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
        description='Export SwiftFormer arch'
    )
    parser.add_argument('--output', type=Path,
                        default=Path('checkpoints/model.onnx'))
    parser.add_argument('--input_size', type=int, nargs='+', default=(224, 224))
    parser.add_argument('--b_size', type=int, default=1, help='Batch size')
    parser.add_argument('--dim', type=int, default=3, help='Channels')
    parser.add_argument('--output_file', action='store_true', default=True,
                        help='Path is a file')
    parser.add_argument('--overwrite', '-f', action='store_true', default=False,
                        help='Overwrite existing onnx')
    parser.add_argument('--verbose', '-v', action='store_true', default=False,
                        help='Verbose Net & ONNX')
    parser.add_argument('--opset_version', type=int, default=11,
                        help='ONNX Opset version')
    parser.add_argument('--arch', default='SwiftFormer_L1', help='config name')
    args = parser.parse_args()

    # Sanitization & verbosity
    if not hasattr(swiftformer, args.arch):
        raise ValueError('Arch not found')

    if args.output_file and args.output.suffix != '.onnx':
        args.output = args.output / 'model.onnx'
    if args.output.exists() and not args.overwrite:
        raise ValueError(f'{args.output=} existed. Set --overwrite / -f')

    main(args)
