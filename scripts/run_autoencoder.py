import torch
from submodules.mask_autoencoder.autoencoder import AE, get_difference_map, flatten, image_size, latent_dim, input_channels

def run_autoencoder(image_mask, video_mask):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AE(latent_dim, input_channels, image_size).to(device)
    model.load_state_dict(torch.load("/workspace/model_weights/autoencoder.pth"))
    model.eval()

    with torch.no_grad():
        image_mask = torch.unsqueeze(torch.Tensor(flatten(image_mask)), 0).to(device)
        n_frames = len(video_mask)
        i = 0
        j = 1

        masks = []
        while j < n_frames:
            input_frame = torch.Tensor(flatten(video_mask[i])).to(device)
            output_frame = torch.Tensor(flatten(video_mask[j])).to(device)
            diff_map =  torch.unsqueeze(get_difference_map(input_frame, output_frame), 0).to(device)
            pred_mask = model(diff_map, image_mask)
            image_mask = pred_mask*255
            masks.append(image_mask.detach().cpu().to(torch.uint8).numpy()[0, :, :])
            i += 1
            j += 1

    return masks
