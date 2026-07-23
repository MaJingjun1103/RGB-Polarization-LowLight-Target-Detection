import torch
from PIL import Image
from torchvision import transforms
import os
from train_hybrid_v1_final import build_hybrid_v1_generator
from models.generator import LightEnhanceGenerator


# =========================
# 路径
# =========================

checkpoint = r"D:\LUNA_Low_Light_Enhancement-main\LUNA_Low_Light_Enhancement-main\checkpoints\hybrid_v1_stage2_best.pth"

input_image = r"C:\Users\ADMIN\Desktop\RGB_W_fusion3.png"

output_image = r"C:\Users\ADMIN\Desktop\RGB_W_fusion3_GAN_result.png"


# =========================
# device
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 加载模型
# =========================



checkpoint_data = torch.load(
    checkpoint,
    map_location=device
)
model = build_hybrid_v1_generator().to(device)
checkpoint_data = torch.load(
    checkpoint,
    map_location=device,
    weights_only=False
)


if "generator" in checkpoint_data:
    state_dict = checkpoint_data["generator"]
else:
    state_dict = checkpoint_data


model.load_state_dict(state_dict)

model.eval()

# 兼容保存格式
if "generator" in checkpoint_data:
    model.load_state_dict(checkpoint_data["generator"])
else:
    model.load_state_dict(checkpoint_data)


model.to(device)
model.eval()


# =========================
# 图像预处理
# =========================

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,0.5,0.5),
        (0.5,0.5,0.5)
    )
])


img = Image.open(input_image).convert("RGB")

w,h = img.size

x = transform(img)
x = x.unsqueeze(0).to(device)



# =========================
# 推理
# =========================

with torch.no_grad():

    y = model(x)

# [-1,1] → [0,1]

y = (y.squeeze(0).cpu()+1)/2

y = torch.clamp(y,0,1)


out = transforms.ToPILImage()(y)

# 恢复原尺寸

out.save(output_image)


print("完成!")
print(output_image)