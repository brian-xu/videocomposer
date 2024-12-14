python pipeline.py \
    --cfg configs/sam_iam_config.yaml \
    --input_text_desc "A tall oak tree's leaves being blown off in a gust of wind."

python run_net.py \
    --cfg configs/sam_iam_config.yaml \
    --input_video "/workspace/outputs/b3563563/sketch.mp4" \
    --image_path "/workspace/submodules/mask_segmentation/output/cropped_input_image.jpeg" \
    --style_image "/workspace/submodules/mask_segmentation/output/cropped_input_image.jpeg" \
    --input_text_desc "An ash tree being uprooted from the ground."