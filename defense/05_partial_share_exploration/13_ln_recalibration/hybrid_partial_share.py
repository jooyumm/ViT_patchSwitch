"""
probes/hybrid_partial_share.py — [탐색적, 롤백 가능] "부분 공유 backbone" 시제품.

배경
----
activation_similarity.py 결과: 독립 학습된 P16/P8 체크포인트가 layer 6~9 시점에는
CKA 0.87~0.96(우연 수준 0.11~0.38 대비 뚜렷이 높음)으로 이미 상당히 정렬된 표현을 쓴다.
이건 diversity diagnostic이 보여준 "토큰화 격자 불일치가 방어의 핵심"이라는 결론과
충돌하지 않는다 — 그 불일치는 patch embedding~초반 층(추정)에서 생기고, layer 6+ 는
어차피 수렴해 있으니 그 구간만 공유해도 방어 기제(초반부)는 안 건드릴 수 있다는 가설.

이 파일은 그 가설을 구조로 구현한 것: patch_embed + block[0:split_layer](기본 5,
"layer 1~5")는 P16/P8 각자 독립 유지, block[split_layer:](기본 "layer 6~12") + norm + head는
하나만 두고 두 경로가 공유한다. 공유되는 부분의 가중치는 P16 체크포인트 것을 그대로
쓴다(P16이 기본 추론 모델이므로).

중요한 한계
-----------
이 공유 후반부는 원래 model16 전체를 학습할 때 생긴 가중치이지, "P8 초반부 출력을 받도록"
재학습/미세조정된 게 전혀 아니다. 즉 이건 진짜 배포 가능한 모델이 아니라, 구조적으로
가능한지 + 이 상태로 joint attack에 얼마나 버티는지를 보는 최소 시제품이다. P8 경로의
clean accuracy가 낮게 나올 수 있고, 그러면 "얼마나 잘 뚫리는지"도 해석에 주의가 필요하다
(이미 망가진 분류기는 공격 없이도 틀리므로). 이 스크립트는 그 clean accuracy를 먼저
측정해서 정직하게 보고한다.

주의: src/models.py는 import만(수정 없음), 원본 model16/model8의 서브모듈(block, norm,
head 등)을 그대로 재사용(같은 파라미터 객체를 참조)할 뿐 새로 학습/수정하지 않는다.
이 파일 지우면 원상복구.
"""
import torch
import torch.nn as nn


class HybridViT(nn.Module):
    """patch_embed+early block은 P16/P8 각자, late block+norm+head는 P16 것을 공유."""

    def __init__(self, model16, model8, split_layer=5):
        super().__init__()
        assert model16.no_embed_class is False and model8.no_embed_class is False, \
            "이 구현은 원본 timm ViT의 표준 pos_embed 경로(concat 후 add)만 지원함"
        assert getattr(model16, 'reg_token', None) is None and getattr(model8, 'reg_token', None) is None
        assert getattr(model16, 'attn_pool', None) is None and getattr(model8, 'attn_pool', None) is None

        self.split_layer = split_layer

        # ── P16 독립 초반부 ──
        self.patch_embed16 = model16.patch_embed
        self.cls_token16 = model16.cls_token
        self.pos_embed16 = model16.pos_embed
        self.pos_drop16 = model16.pos_drop
        self.blocks_early16 = model16.blocks[:split_layer]

        # ── P8 독립 초반부 ──
        self.patch_embed8 = model8.patch_embed
        self.cls_token8 = model8.cls_token
        self.pos_embed8 = model8.pos_embed
        self.pos_drop8 = model8.pos_drop
        self.blocks_early8 = model8.blocks[:split_layer]

        # ── 공유 후반부: model16 것 그대로 (재학습 없음) ──
        self.blocks_late = model16.blocks[split_layer:]
        self.norm = model16.norm
        self.fc_norm = model16.fc_norm   # augreg 계열은 Identity
        self.head = model16.head

    def _early(self, x, patch_embed, cls_token, pos_embed, pos_drop, blocks_early):
        x = patch_embed(x)
        B = x.shape[0]
        cls = cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + pos_embed
        x = pos_drop(x)
        for blk in blocks_early:
            x = blk(x)
        return x

    def forward(self, images, which):
        assert which in ('16', '8')
        if which == '16':
            x = self._early(images, self.patch_embed16, self.cls_token16,
                             self.pos_embed16, self.pos_drop16, self.blocks_early16)
        else:
            x = self._early(images, self.patch_embed8, self.cls_token8,
                             self.pos_embed8, self.pos_drop8, self.blocks_early8)
        for blk in self.blocks_late:
            x = blk(x)
        x = self.norm(x)
        x = x[:, 0]              # global_pool='token' -> CLS
        x = self.fc_norm(x)
        return self.head(x)


class HybridBranch(nn.Module):
    """joint_patch_fool_attack 등 기존 코드가 기대하는 '모델처럼 생긴' 래퍼.
    model(x) 호출 시 hybrid.forward(x, which)를 실행하고, _collect_attn()이 필요로 하는
    .blocks 속성(순서대로 12개 block, .attn 서브모듈 보유)도 제공한다."""

    def __init__(self, hybrid: HybridViT, which: str):
        super().__init__()
        self.hybrid = hybrid   # 서브모듈로 등록 -> zero_grad()가 공유부까지 포함해서 정리됨
        self.which = which
        early = hybrid.blocks_early16 if which == '16' else hybrid.blocks_early8
        # nn.ModuleList로 만들면 파라미터가 이중 등록되므로 일반 list로만 유지
        # (forward에는 안 쓰고 _collect_attn의 hook 등록용 순회에만 쓰임)
        self.blocks = list(early) + list(hybrid.blocks_late)

    def forward(self, x):
        return self.hybrid(x, self.which)


def build_hybrid(model16, model8, split_layer=5):
    model16.eval()
    model8.eval()
    return HybridViT(model16, model8, split_layer=split_layer)


@torch.no_grad()
def sanity_check_branch16_matches_model16(hybrid, model16, images, atol=1e-5):
    """branch16 = early16 + shared late(=model16 것) 이므로 model16(images)와
    100% 동일해야 한다 (같은 파라미터 객체, 같은 연산 순서). 다르면 wiring 버그."""
    branch16 = HybridBranch(hybrid, '16')
    branch16.eval()
    out_branch = branch16(images)
    out_orig = model16(images)
    max_diff = (out_branch - out_orig).abs().max().item()
    ok = torch.allclose(out_branch, out_orig, atol=atol)
    return ok, max_diff
