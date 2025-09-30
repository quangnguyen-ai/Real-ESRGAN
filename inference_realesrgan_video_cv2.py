import argparse
import cv2
import os
from tqdm import tqdm
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact


def main():
    """Video inference using OpenCV instead of ffmpeg."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='Input video path')
    parser.add_argument(
        '-n',
        '--model_name',
        type=str,
        default='realesr-animevideox2v3',
        help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
              'realesr-animevideov3 | realesr-animevideox2v3 | realesr-general-x4v3 | realesr-general-x2v3'))
    parser.add_argument('-o', '--output', type=str, default=None, help='Output video path')
    parser.add_argument('-s', '--outscale', type=float, default=2, help='The final upsampling scale of the image')
    parser.add_argument(
        '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument(
        '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')
    parser.add_argument('--fps', type=float, default=None, help='Output video fps (default: same as input)')
    parser.add_argument('--codec', type=str, default='mp4v', help='Video codec (default: mp4v). Options: mp4v, avc1, h264')

    args = parser.parse_args()

    # Determine models according to model names
    args.model_name = args.model_name.split('.')[0]
    if args.model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
    elif args.model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    elif args.model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
    elif args.model_name == 'realesr-animevideox2v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=2, act_type='prelu')
        netscale = 2
    elif args.model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
    elif args.model_name == 'realesr-general-x2v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=2, act_type='prelu')
        netscale = 2
    else:
        raise ValueError(f'Model {args.model_name} is not supported')

    # Determine model paths
    if args.model_path is not None:
        model_path = args.model_path
    else:
        model_path = os.path.join('weights', args.model_name + '.pth')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f'Model path {model_path} does not exist')

    # Create upsampler
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        gpu_id=args.gpu_id)

    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise ValueError(f'Cannot open video {args.input}')

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_fps = args.fps if args.fps is not None else input_fps

    print(f'Input video: {args.input}')
    print(f'Resolution: {width}x{height}')
    print(f'FPS: {input_fps}')
    print(f'Total frames: {total_frames}')
    print(f'Output scale: {args.outscale}x')

    # Set output path
    if args.output is None:
        video_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f'{video_name}_{args.model_name}.mp4'

    # Create video writer (initially None, will be created after first frame)
    writer = None

    # Process video frame by frame
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc='Processing', unit='frame')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Enhance frame
            try:
                output, _ = upsampler.enhance(frame, outscale=args.outscale)
            except RuntimeError as error:
                print(f'\nError at frame {frame_idx}: {error}')
                print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
                break

            # Initialize writer after first frame (to get output dimensions)
            if writer is None:
                output_height, output_width = output.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*args.codec)
                writer = cv2.VideoWriter(args.output, fourcc, output_fps, (output_width, output_height))
                print(f'Output resolution: {output_width}x{output_height}')
                print(f'Output FPS: {output_fps}')
                print(f'Codec: {args.codec}')

            # Write frame
            writer.write(output)

            frame_idx += 1
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()

    print(f'\n✅ Video processing completed!')
    print(f'Output saved to: {args.output}')


if __name__ == '__main__':
    main()