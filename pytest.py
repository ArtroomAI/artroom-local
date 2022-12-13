try:
    import torch
    if torch.cuda.is_available():
        print("success")
    else:
        print("CUDA could not be loaded. Please check installation.")
except:
    print("Pytorch dependency could not be found. Please check installation  ")
