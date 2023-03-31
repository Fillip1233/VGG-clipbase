import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTextModel, AutoTokenizer
from getgt import AverageMeter,selfatt
from transformers import AutoProcessor, XCLIPVisionModel,XCLIPVisionConfig
from saveit import savepred
import numpy as np
import map

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
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        # self.tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.textfea_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        # Self-Attention
        self.self_attention = selfatt(embed_size=512, nhead=8, dim_feedforward=2048)
        # self.logit_scale = nn.Parameter(torch.ones([]) * self.config.logit_scale_init_value)
        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)
        self.obj_compress = nn.Linear(512, 37)
        
    
    def forward(self,text_input,pixel_values,device):
   
        # vision encode
    
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, num_channels, height, width)
        self.vision_model.to(device)
        vision_outputs = self.vision_model(pixel_values)

        # Encode text 
        self.text_model.to(device)
        outputs1 = self.text_model(**text_input) 
        input_ids = text_input['input_ids']
        position_ids = torch.arange(input_ids.size(1)).expand(input_ids.size(0), -1).to(device)
        last_hidden_state = outputs1.last_hidden_state  
        pooled_output = outputs1.pooler_output 

        # text_outputs
        text_outputs = self.textfea_model.get_text_features(
            input_ids = torch.tensor(text_input['input_ids']),
            attention_mask = torch.tensor(text_input['attention_mask']),
            position_ids= position_ids,
            output_attentions=True,
            output_hidden_states=True,
            return_dict = True,
        )
        
        # pooled_output = text_outputs[1]
        # text_features = self.clip_text_model.text_projection(pooled_output)
        
        image_embeds = vision_outputs[1]
        image_embeds = self.vision_model.visual_projection(image_embeds)

        text_embeds = self.textfea_model.text_projection(text_outputs)
        
        # for k in range(8):
        #     t1=image_embeds[k,:].unsqueeze(0)
        #     t1=torch.cat((t1,text_embeds),dim=0)
        #     in_t1=t1.unsqueeze(0)
        #     out_t1=self.self_attention(in_t1, in_t1, in_t1, None)
        #     image_embeds_1, text_embeds_1 = torch.split(out_t1.squeeze(0), [1, 35], dim=0)
        #     obj_out=self.obj_compress(image_embeds_1[0,:])
            
    
        # Self-Attention
        text_embeds=text_embeds.unsqueeze(0)   #[1,35,512]
        text_embeds=text_embeds.repeat(num_frames,1,1)   #[64,35,512]
        image_embeds=image_embeds.unsqueeze(1)   #[64,1,512]
        input_tensor=torch.cat([image_embeds, text_embeds], dim=1)  #[64,36,512] 

        for i in range(input_tensor.shape[0]):
            tensor = input_tensor[i:i+1]  # 取出第i个张量，形状为 (1, 36, 512)
            # tensor = tensor.transpose(0, 1)  # 将形状变为 (36, 1, 512)，以适应 self-attention 操作的输入格式
            tensor = self.self_attention(tensor)[0]  # 进行 self-attention 操作
            # tensor = tensor.transpose(0, 1)  # 将形状还原为 (1, 36, 512)
            output_tensor = tensor if i == 0 else torch.cat([output_tensor, tensor], dim=0)  # 将每个张量的输出拼接起来
        image_embeds,text_embeds= torch.split(output_tensor, [1, 37], dim=1)   #text_embeds

        # obj_out=self.obj_compress(image_embeds[0,:])
        
        # input_tensor=input_tensor.unsqueeze(0)
        # output_tensor = self.self_attention(input_tensor, input_tensor, input_tensor, None)
        # output_tensor = torch.relu(output_tensor)
        
        for param in self.self_attention.parameters():
            param.requires_grad_(True)
            
        # Normalized Features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        # image_embeds.detach().requires_grad_(False)
        # text_embeds.detach().requires_grad_(False)
        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()

        # logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * self.logit_scale
        # logits_per_text = torch.matmul(torch.squeeze(text_embeds), image_embeds.transpose(1, 2)) * self.logit_scale
        # logits_per_image = logits_per_text.t()      


        logits_per_text = torch.matmul(image_embeds.squeeze(1), text_embeds[0].squeeze(0).t()) *logit_scale  #squeeze从[64,1,512]变成了[64,512]，permute函数交换image_embeds的第二个和第三个维度（从[64,35,512]变成了[64,512,35]）
        # logits_per_image = logits_per_text.permute(0, 2, 1)
        logits_per_image = logits_per_text.t() 
        logits_per_text= torch.sigmoid(logits_per_text)
        # savepred(logits_per_image,1)
        # mAP, _, ap = evamap.charades_map(np.vstack(logits_per_text), np.vstack(obj_gt))
        # print(ap)
        # print(' * mAP {:.3f}'.format(mAP))
        # loss = clip_loss(logits_per_text)
        
        
        return text_input,pixel_values,logits_per_text
        # return text_input,pixel_values



