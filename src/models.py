import torch
import timm

# ViT_tradeoff/src/models.py에서 그대로 가져오되, 이 프로젝트(PatchSwitch)는 P8/P16만
# 다룬다(PGD/P32는 범위 밖) -- P32 항목만 제외
MODEL_NAMES = {
    8:  'vit_base_patch8_224.augreg2_in21k_ft_in1k',
    16: 'vit_base_patch16_224.augreg2_in21k_ft_in1k',
}

def get_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"  Device: Apple MPS")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print(f"  Device: CPU")
    return device


def load_vit_model(patch_size, device):
    name = MODEL_NAMES.get(patch_size, f'vit_base_patch{patch_size}_224')

    try:
        print(f"  시도: {name}")
        model = timm.create_model(name, pretrained=True)

        # patch_size 검증 — timm 모델이 요청한 P와 실제로 다를 경우 스킵
        actual_p = getattr(model.patch_embed, 'patch_size', None)
        if actual_p is not None:
            # timm은 (H, W) tuple 또는 int 둘 다 사용
            actual_p = actual_p[0] if isinstance(actual_p, (tuple, list)) else actual_p
            if actual_p != patch_size:
                print(f"  ✗ patch_size 불일치: 요청={patch_size}, 실제={actual_p} → 스킵")
                return None

        model = model.to(device)
        model.eval()
        num_tokens = (224 // patch_size) ** 2
        print(f"  ✓ 로드 성공: {name}  "
              f"(P={patch_size}, tokens={num_tokens}, "
              f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M)")
        return model

    except Exception as e:
        print(f"  ✗ 실패: {e}")
        return None
