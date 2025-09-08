import torch, time

# CPU
x_cpu = torch.rand(10000, 10000)
start = time.time()
torch.mm(x_cpu, x_cpu)
print("CPU time:", time.time() - start)

# GPU
x_gpu = torch.rand(10000, 10000, device="cuda")
start = time.time()
torch.mm(x_gpu, x_gpu)
torch.cuda.synchronize()
print("GPU time:", time.time() - start)
