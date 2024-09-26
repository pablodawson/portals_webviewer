# MIT Licensed (c) 2024 Christopher Howard
# Adapted from Kevin Kwok's ply2spat.py

import math
import numpy as np
import argparse
import json
from io import BytesIO
import torch
from tqdm import tqdm

HDR_PROTO="""[{"type":"splat","size":30146560,"texwidth":4096,"texheight":460,"cameras":[{"id":0,"img_name":"00001","width":1959,"height":1090,"position":[-3.0089893469241797,-0.11086489695181866,-3.7527640949141428],"rotation":[[0.876134201218856,0.06925962026449776,0.47706599800804744],[-0.04747421839895102,0.9972110940209488,-0.057586739349882114],[-0.4797239414934443,0.027805376500959853,0.8769787916452908]],"fy":1164.6601287484507,"fx":1159.5880733038064}]}]"""
MAGIC=0x674b
C0 = 0.28209479177387814

def SH2RGB(sh):
    return sh * C0 + 0.5

def packHalf2x16(a):

    x=a[np.ix_(*[range(0,i,2) for i in a.shape])]
    y=a[np.ix_(*[range(1,i,2) for i in a.shape])]

    assert (x.shape == y.shape)
    assert (len(x.shape) == 1)

    # Allocate output data
    n = x.shape[0]
    uint32_data = np.ndarray((n,), dtype=np.uint32)

    # Create view
    f16_data = np.lib.stride_tricks.as_strided(
        uint32_data.view(dtype=np.float16),
        shape=(2, n),
        strides=(1 * 2, 2 * 2),
        writeable=True,
    )

    # Convert from whatever type x and y use
    f16_data[0] = x
    f16_data[1] = y

    return uint32_data

def process_ckpt_to_splatv(ckpt_file_path):
    splats = torch.load(ckpt_file_path)["splats"]

    for k, v in splats.items():
        splats[k] = v.cpu().numpy()
    
    scales = splats["scales"]
    opacities = splats["opacities"]

    sorted_indices = np.argsort(
        -np.exp(scales[:,0] + scales[:,1] + scales[:,2])
        / (1 + np.exp(-opacities))
    )
    buffer = BytesIO()

    #BUILD HEADER
    hdr=json.loads(HDR_PROTO)

    vertexCount=len(sorted_indices)
    texwidth= 1024*4
    texheight= math.ceil((4 * vertexCount) / texwidth)
    size=texwidth * texheight * 16

    hdr[0]['texwidth']=texwidth
    hdr[0]['texheight'] = texheight
    hdr[0]['size']= size
    hdrstr=json.dumps(hdr,separators=(',', ':'))

    #write magic
    buffer.write(np.array([MAGIC,len(hdrstr)],dtype=np.uint32).tobytes())
    #write header
    buffer.write(bytes(hdrstr,'utf-8'))

    for idx in tqdm(sorted_indices):

        position = splats["means"][idx].astype(np.float32)
        rot = splats["quats"][idx].astype(np.float32)
        scales = splats["scales"][idx].astype(np.float32)

        rotscales = np.array(
            [rot[0], rot[1], rot[2], rot[3],
                    math.exp(scales[0]), math.exp(scales[1]), math.exp(scales[2]),0.0],
                  dtype=np.float32,
        )

        dc = SH2RGB(splats["sh0"][idx].astype(np.float32)[0])
        opacity = splats["opacities"][idx].astype(np.float32)
        motion = splats["motion"][idx].astype(np.float32)
        omega = splats["omega"][idx].astype(np.float32)
        trbf_scale = splats["exp_scale"][idx].astype(np.float32)
        trbf_center = splats["time_center"][idx].astype(np.float32)[0]

        fdc = np.array(
                [max(0, min(255, dc[0] * 255)),
                 max(0, min(255, dc[1] * 255)),
                 max(0, min(255, dc[2] * 255)),
                 (1 / (1 + math.exp(-opacity))) * 255],
            dtype=np.uint8,
        )

        motion = np.array(
            [motion[0], motion[1], motion[2],motion[3],
                   motion[4], motion[5], motion[6],motion[7],
                   motion[8],  0.0, omega[0] ,omega[1] ,
                   omega[2], omega[3], trbf_center,math.exp(trbf_scale)],
            dtype=np.float32,
        )

        #asdaf

        buffer.write(position.tobytes())
        buffer.write(packHalf2x16(rotscales).tobytes())
        buffer.write(fdc.tobytes())
        buffer.write(packHalf2x16(motion).tobytes())

    return buffer.getvalue()


def save_splat_file(splat_data, output_path):
    with open(output_path, "wb") as f:
        f.write(splat_data)


def main():
    parser = argparse.ArgumentParser(description="Convert ckpt files to SPLATV format.")
    parser.add_argument(
        "input_files", nargs="+", help="The input ckpt files to process."
    )
    parser.add_argument(
        "--output", "-o", default="output.splatv", help="The output SPLATV file."
    )
    args = parser.parse_args()
    for input_file in args.input_files:
        print(f"Processing {input_file}...")
        splat_data = process_ckpt_to_splatv(input_file)
        output_file = (
            args.output if len(args.input_files) == 1 else input_file + ".splatv"
        )
        save_splat_file(splat_data, output_file)
        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
