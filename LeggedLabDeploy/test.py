import torch
policy = torch.jit.load("/home/lyf/code/geoloco/LeggedLabDeploy/policy/g1/exported/policy.pt")
dummy_input = torch.zeros(1, 960)  # 96 × 10 = 960
output = policy(dummy_input)
print(f"Input: {dummy_input.shape}, Output: {output.shape}")