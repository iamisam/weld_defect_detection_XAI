import torch
device="cuda"

size=32768  # try max; drop if OOM
a=torch.randn((size,size),device=device)
b=torch.randn((size,size),device=device)

torch.backends.cuda.matmul.allow_tf32 = True

while True:
    for _ in range(20):  # queue more work
        c = a @ b
        a = b @ c
        b = c @ a