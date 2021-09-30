def empty_cache():
    import os
    if 'CUDA_EMPTY_CACHE' in os.environ and int(os.environ['CUDA_EMPTY_CACHE']):
        import torch
        torch.cuda.empty_cache()
