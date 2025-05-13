
import lpips
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as vT
from transformers import AutoProcessor, CLIPModel
from tqdm.auto import tqdm

from .dino_extractor import VitExtractor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LossG(torch.nn.Module):
    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            cfg = {
                "dino_model_name": "dino_vitb8",
                "dino_global_patch_size": 224,
                "clip_model_id": "openai/clip-vit-base-patch32",
            }

        self.extractor = VitExtractor(model_name=cfg["dino_model_name"],device=device)
        imagenet_norm = vT.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        global_resize_transform = vT.Resize(cfg["dino_global_patch_size"], max_size=480)
        self.global_transform = vT.Compose([
            vT.ToTensor(),
            global_resize_transform,
            imagenet_norm
        ])

        self.clip_model = CLIPModel.from_pretrained(cfg["clip_model_id"]).to(device)
        self.clip_processor = AutoProcessor.from_pretrained(cfg["clip_model_id"])
        self.lpips_model = lpips.LPIPS(net="alex").to(device)

    @torch.no_grad()
    def calculate_clip_text_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in tqdm(zip(inputs, outputs), desc="CLIP", total=len(inputs), leave=False):  # a is text, b is image
            with torch.no_grad():
                inputs = self.clip_processor(
                    text = [a],
                    images = b,
                    return_tensors = "pt",
                ).to(device)

                outputs = self.clip_model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                # Compute the similarity with cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                similarity = (text_features @ image_features.T).squeeze(0).detach()[0]
                loss += similarity
                
        return loss.item()

    @torch.no_grad()
    def calculate_self_sim_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in tqdm(zip(inputs, outputs), desc="DINO self-similarity", total=len(inputs), leave=False):
            a = self.global_transform(a).to(device)
            b = self.global_transform(b).to(device)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
                keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
                loss += F.mse_loss(keys_ssim, target_keys_self_sim).detach()
        return loss.item()

    @torch.no_grad()
    def calculate_dino_i_loss(self, outputs, inputs):
        loss = 0.0
        for a, b in tqdm(zip(outputs, inputs), desc="DINO-I (CLS cosine similarity)", total=len(inputs), leave=False):
            a = self.global_transform(a).unsqueeze(0).to(device)
            b = self.global_transform(b).unsqueeze(0).to(device)
            with torch.no_grad():
                cls_token = self.extractor.get_feature_from_input(a)[-1][0, 0, :]
                target_cls_token = self.extractor.get_feature_from_input(b)[-1][0, 0, :]
                loss += F.cosine_similarity(cls_token, target_cls_token, dim=0)
        return loss.item()
    
    @torch.no_grad()
    def calculate_LPIPS_distance(self, outputs, inputs):
        distance = 0.0
        transform = vT.Compose([vT.ToTensor()])
        for a, b in tqdm(zip(outputs, inputs), desc="LPIPS", total=len(inputs), leave=False):
            a = transform(a).unsqueeze(0).to(device)
            b = transform(b).unsqueeze(0).to(device)
            with torch.no_grad():
                distance += self.lpips_model(a, b).detach()
        return distance.item()
