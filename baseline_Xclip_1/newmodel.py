import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTextModel, AutoTokenizer
from getgt import AverageMeter,selfatt
from transformers import AutoProcessor, XCLIPVisionModel,XCLIPVisionConfig
from saveit import savepred
import numpy as np

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def mulit_loss(pred,gt):
    mlm_loss = nn.MultiLabelMarginLoss()
    loss = mlm_loss(pred,gt)
    return loss


class base_Model(nn.Module):
    def __init__(self):
        super(base_Model, self).__init__()
        
       # initialize vision model
        self.configuration = XCLIPVisionConfig()
        # self.processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        self.vision_model = XCLIPVisionModel(self.configuration)
        self.configuration = self.vision_model.config

        # initialize text model
        # self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.textfea_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # Self-Attention
        self.self_attention = selfatt(embed_size=512, nhead=8, dim_feedforward=2048)
        # self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592) ##hyper-parameter
        self.obj_compress = nn.Linear(512, 37)
        self.visual_fc = nn.Linear(768, 512, bias=False)
        self.text_fc = nn.Linear(512, 512, bias=False)
    
    def forward(self,text_input,pixel_values,device):
   
        # vision encode
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)
        # self.vision_model.to(device)
        # _ = torch.set_grad_enabled(False)
        vision_outputs = self.vision_model(pixel_values)

        # Encode text 
        # self.text_model.to(device)
        input_ids = text_input['input_ids']
        position_ids = torch.arange(input_ids.size(1)).expand(input_ids.size(0), -1).to(device)

        # text_outputs
        text_outputs = self.textfea_model.get_text_features(
            input_ids = torch.tensor(text_input['input_ids']),
            attention_mask = torch.tensor(text_input['attention_mask']),
            position_ids= position_ids,
            output_attentions=True,
            output_hidden_states=True,
            return_dict = True,
        )
        
        image_embeds = vision_outputs[1]
        # _ = torch.set_grad_enabled(True)
        image_embeds = self.visual_fc(image_embeds)
        text_embeds = self.text_fc(text_outputs)
    
        # Self-Attention
        # text_embeds = text_embeds.unsqueeze(0)   #[1,37,512]
        # text_embeds = text_embeds.repeat(num_frames*batch_size,1,1)   #[bs*n_frame,37,512]
        # image_embeds = image_embeds.unsqueeze(1)   #[bs*n_frame,1,512]
        # input_tensor = torch.cat([image_embeds, text_embeds], dim=1)  #[bs*n_frame,38,512] 

        # tensor = self.self_attention(input_tensor)[0]
        # image_embeds,text_embeds= torch.split(tensor, [1, 37], dim=1)   #text_embeds
        # b = image_embeds.squeeze(1) 

        # Normalized Features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale
        # logits_per_text = torch.matmul(torch.squeeze(text_embeds), image_embeds.transpose(1, 2)) * self.logit_scale
        # logits_per_image = logits_per_text.t()      
        logits_per_text=logits_per_text.t()
        # logits_per_text = torch.matmul(image_embeds.squeeze(1), text_embeds[0].squeeze(0).t()) *logit_scale  #squeeze从[64,1,512]变成了[64,512]，permute函数交换image_embeds的第二个和第三个维度（从[64,35,512]变成了[64,512,35]）
        # logits_per_text= torch.sigmoid(logits_per_text)

        

        return text_input, pixel_values, logits_per_text
        # return text_input,pixel_values

# if __name__ == '__main__':
#     from newmodel import base_Model
#     text1_input = ["a photo of a background","a photo of a person","a photo of a bag", "a photo of a bed","a photo of a blanket","a photo of a book","a photo of a box",
#             "a photo of a broom","a photo of a chair","a photo of a closetcabinet","a photo of a clothes","a photo of a cupglassbottle",
#             "a photo of a dish","a photo of a door","a photo of a doorknob","a photo of a doorway","a photo of a floor",
#             "a photo of a food","a photo of a groceries","a photo of a laptop","a photo of a light","a photo of a medicine",
#             "a photo of a mirror","a photo of a papernotebook","a photo of a phonecamera","a photo of a picture","a photo of a pillow",
#             "a photo of a refrigerator","a photo of a sandwich","a photo of a shelf","a photo of a shoe","a photo of a sofacouch",
#             "a photo of a table","a photo of a television","a photo of a towel","a photo of a vacuum","a photo of a window"]
#     model = base_Model()
#     bs = 2
#     n_frame = 8
#     pixel_values = torch.Tensor(bs, n_frame, 3, 224, 224)
#     text_input = torch.Tensor(bs, n_frame, 3, 224, 224)
#     device = torch.device("cpu")
#     tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
#     text_input = tokenizer(text1_input, padding=True, return_tensors="pt").to(device)
#     output = model(text_input, pixel_values, device)

