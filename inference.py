import argparse
from bg_remove import BGRemove

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_image', type=str, default='pretrained/modnet_photographic_portrait_matting.ckpt',
                        required=False, help='Checkpoint path')
    parser.add_argument('--ckpt_video', type=str, default='pretrained/modnet_webcam_portrait_matting.ckpt',
                        required=False, help='Checkpoint path')
    parser.add_argument('--image', type=str, default='',
                        required=False, help='Inference image filename')
    parser.add_argument('--video', type=str, default='',
                        required=False, help='Inference image filename')
    parser.add_argument('--webcam', type=bool, default=False,
                        required=False, help='Realtime webcam')
    parser.add_argument('--folder', type=str, default='assets/sample_image',
                        required=False, help='Inference images foldername')
    parser.add_argument('--background', type=bool, default=False,
                        required=False, help='Background image adding')

    args = parser.parse_args()
    try:
        if args.webcam or args.video:
            bg_remover = BGRemove(args.ckpt_video)
        else:
            bg_remover = BGRemove(args.ckpt_image)

        if args.image:
            bg_remover.image(args.image, background=args.background)
        elif args.video:
            bg_remover.video(args.video, background=args.background)
        elif args.webcam:
            bg_remover.webcam(background=args.background)
        else:
            bg_remover.folder(args.folder, background=args.background)

    except Exception as Err:
        print("Erro happend {}".format(Err))
