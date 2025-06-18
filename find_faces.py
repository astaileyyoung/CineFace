import time 
import subprocess as sp
from pathlib import Path 

from argparse import ArgumentParser


def find_faces(src, dst=None, image='astaileyyoung/visage', frameskip=24):
    src = Path(src).absolute().resolve()
    dst = Path(dst).absolute().resolve() if dst else None
    mount_point_src = src.parent
    mount_point_dst = dst.parent if dst else None
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=all",
        "-v",
        f"{str(mount_point_src)}:/app/{mount_point_src.parts[-1]}"
    ]
    if dst:
        command.extend([
            "-v",
            f"{str(mount_point_dst)}:/app/{mount_point_dst.parts[-1]}"
        ])
    command.extend([
        image,
        "visage",
        str(Path(mount_point_src.parts[-1]).joinpath(src.name)),
        "--frameskip",
        str(frameskip)
    ])
    if dst:
        command.append('--dst')
        command.append(str(Path(mount_point_dst.parts[-1]).joinpath(dst.name)))
        print(command)
    sp.run(command)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--dst')
    ap.add_argument('--frameskip', default=24, type=int)
    ap.add_argument('--image', default='astaileyyoung/visage', type=str)
    args = ap.parse_args()
    find_faces(args.src, dst=args.dst, image=args.image, frameskip=args.frameskip)
