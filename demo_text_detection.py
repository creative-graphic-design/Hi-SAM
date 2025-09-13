import argparse
import glob
import os
import random
import warnings

import cv2
import numpy as np
import torch
from tqdm import tqdm

from hi_sam.demo.config_utils import get_detection_params
from hi_sam.demo.mask_utils import create_binary_mask
from hi_sam.demo.visualization import show_masks
from hi_sam.modeling.auto_mask_generator import AutoMaskGenerator
from hi_sam.modeling.build import model_registry

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser("Hi-SAM", add_help=False)

    parser.add_argument(
        "--input", type=str, required=True, nargs="+", help="Path to the input image"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./demo",
        help="A file or directory to save output visualizations.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_l",
        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="The device to run generation on."
    )
    parser.add_argument("--hier_det", default=True)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="totaltext",
        help="'totaltext' or 'ctw1500', or 'ic15'.",
    )
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--zero_shot", action="store_true")
    parser.add_argument("--save_mask", action="store_true", help="Save binary mask image (white text, black background)")

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--input_size", default=[1024, 1024], type=list)

    # self-prompting
    parser.add_argument(
        "--attn_layers",
        default=1,
        type=int,
        help="The number of image to token cross attention layers in model_aligner",
    )
    parser.add_argument(
        "--prompt_len", default=12, type=int, help="The number of prompt token"
    )
    parser.add_argument("--layout_thresh", type=float, default=0.5)
    return parser.parse_args()




if __name__ == "__main__":
    args = get_args_parser()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hisam = model_registry[args.model_type](args)
    hisam.eval()
    hisam.to(args.device)
    print("Loaded model")
    amg = AutoMaskGenerator(hisam)

    params = get_detection_params(args.dataset, args.zero_shot)
    fg_points_num = params["fg_points_num"]
    score_thresh = params["score_thresh"]

    if os.path.isdir(args.input[0]):
        args.input = [
            os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])
        ]
    elif len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
    for path in tqdm(args.input):
        img_id = os.path.basename(path).split(".")[0]

        if os.path.isdir(args.output):
            assert os.path.isdir(args.output), args.output
            img_name = os.path.basename(path).split(".")[0] + ".png"
            out_filename = os.path.join(args.output, img_name)
            mask_filename = os.path.join(args.output, os.path.basename(path).split(".")[0] + "_mask.png")
        else:
            assert len(args.input) == 1
            out_filename = args.output
            mask_filename = os.path.splitext(args.output)[0] + "_mask.png"

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # h, w, 3
        img_h, img_w = image.shape[:2]

        amg.set_image(image)
        masks, scores = amg.predict_text_detection(
            from_low_res=False,
            fg_points_num=fg_points_num,
            batch_points_num=min(fg_points_num, 100),
            score_thresh=score_thresh,
            nms_thresh=score_thresh,
            zero_shot=args.zero_shot,
            dataset=args.dataset,
        )

        if masks is not None:
            print("Inference done. Start plotting masks.")
            show_masks(masks, out_filename, image)

            if args.save_mask:
                binary_mask = create_binary_mask(masks, image.shape)
                cv2.imwrite(mask_filename, binary_mask)
                print(f"Binary mask saved to: {mask_filename}")
        else:
            print("no prediction")
