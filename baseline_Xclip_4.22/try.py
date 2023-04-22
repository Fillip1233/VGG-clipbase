from PIL import Image
import requests
from transformers import AutoProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

# import av
# import torch
# import numpy as np

# from transformers import AutoProcessor, AutoModel
# from huggingface_hub import hf_hub_download

# np.random.seed(0)


# def read_video_pyav(container, indices):
#     '''
#     Decode the video with PyAV decoder.
#     Args:
#         container (`av.container.input.InputContainer`): PyAV container.
#         indices (`List[int]`): List of frame indices to decode.
#     Returns:
#         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
#     '''
#     frames = []
#     container.seek(0)
#     start_index = indices[0]
#     end_index = indices[-1]
#     for i, frame in enumerate(container.decode(video=0)):
#         if i > end_index:
#             break
#         if i >= start_index and i in indices:
#             frames.append(frame)
#     return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
#     converted_len = int(clip_len * frame_sample_rate)
#     end_idx = np.random.randint(converted_len, seg_len)
#     start_idx = end_idx - converted_len
#     indices = np.linspace(start_idx, end_idx, num=clip_len)
#     indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
#     return indices


# # video clip consists of 300 frames (10 seconds at 30 FPS)
# file_path = hf_hub_download(
#     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
# )
# container = av.open(file_path)

# # sample 8 frames
# indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
# video = read_video_pyav(container, indices)

# processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
# model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

# inputs = processor(
#     text=["playing sports", "eating spaghetti", "go shopping"],
#     videos=list(video),
#     return_tensors="pt",
#     padding=True,
# )

# # forward pass
# with torch.no_grad():
#     outputs = model(**inputs)

# logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
# probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities
# print(probs)