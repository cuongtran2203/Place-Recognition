from ConTNet import build_model,create_ConTNet_M
import torch
model = build_model(arch='ConT-M', use_avgdown=True, relative=True, qkv_bias=True, pre_norm=True)
model.load_state_dict(torch.load("checkpoint_M.pth"),strict=False)
# Chọn 9 khối đầu tiên
selected_layers = list(model.children())[:-2]

# Tạo một mô hình mới chỉ chứa 9 khối đầu tiên
new_model = torch.nn.Sequential(*selected_layers)
input = torch.Tensor(4, 3, 224, 224)
print(new_model)
out = new_model(input)
print(out.shape)
