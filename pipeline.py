from scripts.run_autoencoder import run_autoencoder
import uuid, os
import cv2
from utils.config import Config

from tools.videocomposer.inference_single import inference_single

cfg = Config(load=True)
cfg.cfg_dict["style_image"] = cfg.cfg_dict["image_path"]

experiment_name = str(uuid.uuid4())[:8]
experiment_path = os.path.join(os.getcwd(), "outputs", experiment_name)
os.mkdir(experiment_path)

print("Intermediate outputs will be stored at", experiment_path)

# fill these in
# masks = run_autoencoder(image_mask, video_mask)

# TODO: replace test code below
paths = []
for entry in os.scandir("/workspace/DAVIS/Annotations/480p/bear"):
    paths.append(entry.path)

paths.sort()
video_mask = []
for path in paths:
    if path[-4:] == ".png":
        img = cv2.imread(path)
        video_mask.append(img)
masks = run_autoencoder(video_mask[0], video_mask)

edges = []
for mask in masks:
    edge = cv2.Canny(mask, 1, 255)
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