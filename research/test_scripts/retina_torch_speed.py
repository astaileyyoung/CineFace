from argparse import ArgumentParser 

import cv2 
import numpy as np
from tqdm import tqdm 

import torch 
import torch.backends.cudnn as cudnn

from utils import resize_image, resize_tensor

from videotools.detectors.RetinaTorch import (
    RetinaFace, load_model, cfg_re50, cfg_mnet, parse_predictions, process_image)


def main(args):
    d = {'cfg_re50': (cfg_re50, '/home/amos/programs/videotools/videotools/detectors/data/Resnet50_Final.pth'),
         'cfg_mnet': (cfg_mnet, '/home/amos/programs/videotools/videotools/detectors/data/Resnet50_Final.pth')}
    torch.set_grad_enabled(False)
    net = RetinaFace(cfg=d[args.model][0])
    model = load_model(net, d[args.model][1], False)
    model = torch.quantization.quantize_dynamic(
    model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8
)
    # if args.float == '16':
    #     model = model.half()
    _ = model.eval()

    cudnn.benchmark = True
    device = torch.device('cuda')
    model = model.to(device)

    sources = {'1': '/home/amos/datasets/test_videos/shining_bat.mp4',
                '2': '/home/amos/media/tv/a_murder_at_the_end_of_the_world/A.Murder.at.the.End.of.the.World.S01E01.2160p.WEB.H265-SuccessfulCrab/a.murder.at.the.end.of.the.world.s01e01.2160p.web.h265-successfulcrab.mkv',
                '3': '/home/amos/media/tv/Mythic.Quest.S03.1080p.ATVP.WEB-DL.DDP5.1.H.264-CasStudio/Mythic.Quest.S03E01.Across.the.Universe.1080p.ATVP.WEB-DL.DDP5.1.H.264-CasStudio.mkv'}
    src = sources[args.src]
    cap = cv2.VideoCapture(src)
    framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for frame_num in tqdm(range(framecount)):
        ret, frame = cap.read()
        if frame_num % 24 == 0:
            frame = resize_image(frame, model='cv2')
            frames.append(frame)
        if len(frames) >= args.batch_size:
            tensors = [process_image(x, float=args.float) for x in frames]
            tensor = torch.cat(tensors)        
            tensor = tensor.to(device)
            # tensor = resize_tensor(tensor)
            h, w = tensor.size()[2:]
            scale = torch.Tensor([w, h, w, h])
            scale = scale.to(device)
            locs, confs, landmss = model(tensor)
            parse_predictions(frames, tensor, scale, 1, locs, confs, landmss, device, h, w) 
            frames = []
            


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--batch_size', default=4, type=int)
    ap.add_argument('--float', default='32', type=str)
    ap.add_argument('--resize', default='cv2')
    ap.add_argument('--src', default='1')
    ap.add_argument('--model', default='cfg_re50')
    args = ap.parse_args()
    main(args)
