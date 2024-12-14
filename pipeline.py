from scripts.calculate_bbox_movement import translate_bbox
import uuid, os
import cv2
from PIL import Image
from utils.config import Config

from tools.videocomposer.inference_single import inference_single

cfg = Config(load=True)
sam_output_dir = "/workspace/submodules/mask_segmentation/output"
input_image = os.path.join(sam_output_dir, "cropped_input_image.jpeg")

cfg.cfg_dict["image_path"] = input_image
cfg.cfg_dict["style_image"] = input_image

experiment_name = str(uuid.uuid4())[:8]
experiment_path = os.path.join(os.getcwd(), "outputs", experiment_name)
os.mkdir(experiment_path)

print("Intermediate outputs will be stored at", experiment_path)

paths = []
# for entry in os.scandir(os.path.join(sam_output_dir, "segmentation_output")):
for entry in os.scandir("/workspace/DAVIS/Annotations/480p/rollerblade"):
    paths.append(entry.path)

paths.sort()
image_mask = Image.open(os.path.join(sam_output_dir, "input_image_mask.jpeg"))
video_mask = []
for path in paths:
    if path[-4:] != ".png":
        continue
    img = Image.open(path)
    video_mask.append(img)
masks = translate_bbox(image_mask, video_mask)

edges = []
for mask in masks:
    edge = cv2.Canny(mask, 1, 200)
    edges.append(edge)

height, width = edges[0].shape

sketch_video_path = os.path.join(experiment_path, "sketch.mp4")

out = cv2.VideoWriter(sketch_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height), isColor=False)
for edge in edges:
    out.write(edge)
out.release()

cfg.cfg_dict["input_video"] = sketch_video_path

print("Beginning diffusion:")

# inference_single(cfg.cfg_dict)