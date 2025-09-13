import io
from typing import List, Literal

import cv2
import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, field_validator

from hi_sam.demo.config_utils import get_detection_params
from hi_sam.demo.mask_utils import create_binary_mask
from hi_sam.demo.visualization import create_masks_image
from hi_sam.modeling.auto_mask_generator import AutoMaskGenerator
from hi_sam.modeling.build import model_registry

app = FastAPI(title="Hi-SAM Text Detection API", version="1.0.0")


class TextDetectionConfig(BaseModel):
    model_type: str = "vit_l"
    checkpoint: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dataset: str = "totaltext"
    zero_shot: bool = False
    save_mask: bool = False
    seed: int = 42
    input_size: List[int] = [1024, 1024]
    attn_layers: int = 1
    prompt_len: int = 12
    layout_thresh: float = 0.5

    @field_validator("checkpoint", mode="before")
    def validate_checkpoint(cls, checkpoint_filename: str):
        checkpoint_path = hf_hub_download(
            repo_id="creative-graphic-design/hi-sam-checkpoints",
            filename=checkpoint_filename,
        )
        return checkpoint_path


class Args:
    def __init__(self, config: TextDetectionConfig):
        self.model_type = config.model_type
        self.checkpoint = config.checkpoint
        self.device = config.device
        self.hier_det = True
        self.dataset = config.dataset
        self.zero_shot = config.zero_shot
        self.save_mask = config.save_mask
        self.seed = config.seed
        self.input_size = config.input_size
        self.attn_layers = config.attn_layers
        self.prompt_len = config.prompt_len
        self.layout_thresh = config.layout_thresh


class ModelService:
    def __init__(self):
        self.model = None
        self.amg = None
        self.current_config = None

    def initialize_model(self, config: TextDetectionConfig):
        if self.current_config is None or self.current_config != config:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)

            args = Args(config)
            self.model = model_registry[config.model_type](args)
            self.model.eval()
            self.model.to(config.device)
            self.amg = AutoMaskGenerator(self.model)
            self.current_config = config

    def get_amg(self):
        return self.amg


@app.get("/")
async def root():
    return {"message": "Hi-SAM Text Detection API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


def get_model_service() -> ModelService:
    if not hasattr(get_model_service, "_instance"):
        get_model_service._instance = ModelService()
    return get_model_service._instance


@app.post("/detect-text")
async def detect_text(
    file: UploadFile,
    checkpoint: Literal[
        "line_detection_ctw1500.pth",
        "word_detection_totaltext.pth",
    ] = "line_detection_ctw1500.pth",
    model_type: Literal[
        "vit_h",
        "vit_l",
        "vit_b",
    ] = "vit_h",
    dataset: Literal[
        "totaltext",
        "ctw1500",
        "ic15",
    ] = "ctw1500",
    zero_shot: bool = False,
    save_mask: bool = False,
    service: ModelService = Depends(get_model_service),
):
    # Create config
    config = TextDetectionConfig(
        model_type=model_type,
        checkpoint=checkpoint,
        dataset=dataset,
        zero_shot=zero_shot,
        save_mask=save_mask,
    )

    # Initialize model if needed
    service.initialize_model(config)

    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w = image.shape[:2]

    # Set detection parameters based on dataset and mode
    try:
        params = get_detection_params(dataset, zero_shot)
        fg_points_num = params["fg_points_num"]
        score_thresh = params["score_thresh"]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Run text detection
    amg = service.get_amg()
    amg.set_image(image)
    masks, scores = amg.predict_text_detection(
        from_low_res=False,
        fg_points_num=fg_points_num,
        batch_points_num=min(fg_points_num, 100),
        score_thresh=score_thresh,
        nms_thresh=score_thresh,
        zero_shot=zero_shot,
        dataset=dataset,
    )

    if masks is None:
        # Return empty image with metadata in headers
        image_buffer = io.BytesIO()
        cv2_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, encoded_image = cv2.imencode(".png", cv2_image)
        image_buffer.write(encoded_image.tobytes())
        image_buffer.seek(0)

        headers = {
            "X-Success": "true",
            "X-Num-Masks": "0",
            "X-Message": "No text detected in the image",
        }
        return Response(
            image_buffer.getvalue(), media_type="image/png", headers=headers
        )

    # Create visualization image with masks
    if save_mask:
        # Return binary mask if requested
        binary_mask = create_binary_mask(masks, image.shape)
        image_buffer = io.BytesIO()
        _, encoded_image = cv2.imencode(".png", binary_mask)
        image_buffer.write(encoded_image.tobytes())
        image_buffer.seek(0)
    else:
        # Return visualization with colored masks overlaid
        image_buffer = create_masks_image(masks, image)

    headers = {
        "X-Success": "true",
        "X-Num-Masks": str(len(masks)),
        "X-Message": f"Successfully detected {len(masks)} text regions",
    }

    return Response(image_buffer.getvalue(), media_type="image/png", headers=headers)
