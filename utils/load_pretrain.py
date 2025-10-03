import torch

def load_checkpoint_partial(model, checkpoint_path, exclude_prefixes=('text_encoder',), strict=False):
    """
    安全加载 checkpoint，自动处理 DDP 的 'module.' 前缀
    支持从 DDP 保存的权重加载到 DDP 或 单卡模型
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    ckpt_state_dict = checkpoint['net']

    model_state_dict = model.state_dict()

    # 判断模型是否为 DDP
    is_ddp_model = list(model_state_dict.keys())[0].startswith('module.')
    
    # 判断 checkpoint 是否为 DDP 保存的
    is_ddp_ckpt = list(ckpt_state_dict.keys())[0].startswith('module.')

    # 统一格式：将 ckpt 转换为和 model 一样的命名风格
    adjusted_ckpt = {}
    for k, v in ckpt_state_dict.items():
        # 跳过要排除的模块（如 text_encoder）
        clean_k = k[7:] if k.startswith('module.') else k
        if any(clean_k.startswith(prefix) for prefix in exclude_prefixes):
            continue

        # 根据当前模型结构调整 key
        if is_ddp_model and not k.startswith('module.'):
            new_key = f'module.{k}'
        elif not is_ddp_model and k.startswith('module.'):
            new_key = k[7:]
        else:
            new_key = k

        if new_key in model_state_dict:
            if v.shape == model_state_dict[new_key].shape:
                adjusted_ckpt[new_key] = v
            else:
                print(f"[Warning] Shape mismatch for {new_key}: {v.shape} vs {model_state_dict[new_key].shape}")
        else:
            print(f"[Warning] Unexpected key after adjustment: {new_key}")

    # 更新并加载
    model_state_dict.update(adjusted_ckpt)
    model.load_state_dict(model_state_dict)

    print(f"Successfully loaded {len(adjusted_ckpt)} parameters from {checkpoint_path}")
    return checkpoint  # 可用于恢复 optimizer 等


def save_checkpoint(model, optimizer, scheduler, epoch, path, exclude_prefixes=('text_encoder',)):
    """
    保存模型 checkpoint，自动处理 DDP 的 'module.' 前缀，并排除指定模块
    """
    # 获取 state_dict
    state_dict = model.state_dict()

    # 如果是 DDP，去掉 'module.' 前缀后再过滤
    filtered_state_dict = {}
    for k, v in state_dict.items():
        # 去掉 module. 前缀（如果是 DDP）
        name = k[7:] if k.startswith('module.') else k
        
        # 跳过要排除的模块（如 text_encoder）
        if not any(name.startswith(prefix) for prefix in exclude_prefixes):
            filtered_state_dict[k] = v  # 保留原始 key（DDP 保存仍带 module.）

    checkpoint = {
        'net': filtered_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)

def load(model, model_dict):
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    elif "network_weights" in model_dict.keys():
        state_dict = model_dict["network_weights"]
    elif "net" in model_dict.keys():
        state_dict = model_dict["net"]
    elif "student" in model_dict.keys():
        state_dict = model_dict["student"]
    else:
        state_dict = model_dict

    if "module." in list(state_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)

    if "backbone." in list(state_dict.keys())[0]:
        print("Tag 'backbone.' found in state dict - fixing!")
    for key in list(state_dict.keys()):
        state_dict[key.replace("backbone.", "")] = state_dict.pop(key)

    if "swin_vit" in list(state_dict.keys())[0]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)

    current_model_dict = model.state_dict()

    count_right, count_all = 0, 0
    for k in current_model_dict.keys():
        count_all += 1
        if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()):
            print(f"correct {k}")
            count_right += 1
        else:
            print(f"mismatch {k}")

    print(f"count_right is {count_right} count_all is {count_all}!")
    print(f"the pretraining parameter match rate is {count_right/count_all}")
    new_state_dict = {
        k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else current_model_dict[k]
        for k in current_model_dict.keys()}

    model.load_state_dict(new_state_dict, strict=True)

    return model