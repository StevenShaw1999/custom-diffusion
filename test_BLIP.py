from lavis.models import load_model_and_preprocess
from PIL import Image
import torch



device = 'cuda:0'
img = Image.open('/data1/jiayu_xiao/project/custom-diffusion/data/barn_single/4/jonah-brown-eYAsBnNISa4-unsplash.jpg').convert('RGB').resize((512,512), Image.Resampling.LANCZOS)
model_blip, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=torch.device(device))
_image = vis_processors["eval"](img).unsqueeze(0)[:, :3, :, :].to(device)
prompt_str = model_blip.generate({"image": _image})[0]
print(prompt_str)
