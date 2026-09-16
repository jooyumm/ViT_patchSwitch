"""
여러 개의 기존 결과 그림(PNG)을 하나의 그리드 그림으로 합치는 공용 유틸리티.
5개 섹션 통합 재구성(2026-09-16)의 일부 -- 각 섹션 폴더에서
`python ../compose_grid.py <출력경로> <라벨1>:<이미지1경로> <라벨2>:<이미지2경로> ...`
형태로 호출한다. 개별 실험의 원본 그림은 손대지 않고 그대로 두고,
그 그림들을 (a)(b)(c)... 패널로 나란히 배치한 합본만 새로 만든다.
"""
import sys
from PIL import Image, ImageDraw, ImageFont

def load_font(size):
    for path in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()

def compose(out_path, items, ncols, label_h=44, pad=14, bg=(255, 255, 255)):
    # items: list of (label, image_path)
    imgs = [Image.open(p).convert('RGB') for _, p in items]
    labels = [lab for lab, _ in items]
    nrows = (len(imgs) + ncols - 1) // ncols

    col_w = max(im.width for im in imgs)
    row_h = max(im.height for im in imgs)
    cell_w, cell_h = col_w + pad, row_h + label_h + pad

    canvas = Image.new('RGB', (cell_w * ncols + pad, cell_h * nrows + pad), bg)
    draw = ImageDraw.Draw(canvas)
    font = load_font(28)

    for idx, (im, lab) in enumerate(zip(imgs, labels)):
        r, c = divmod(idx, ncols)
        x0 = pad + c * cell_w
        y0 = pad + r * cell_h
        draw.text((x0, y0), lab, fill=(0, 0, 0), font=font)
        # 이미지가 셀보다 작으면 가운데 정렬
        ox = x0 + (col_w - im.width) // 2
        oy = y0 + label_h
        canvas.paste(im, (ox, oy))

    canvas.save(out_path)
    print(f"Saved: {out_path}  ({len(imgs)} panels, {ncols}x{nrows})")

if __name__ == '__main__':
    out_path = sys.argv[1]
    items = []
    for arg in sys.argv[2:]:
        if arg.startswith('--ncols='):
            continue
        lab, path = arg.split(':', 1)
        items.append((lab, path))
    ncols = 3
    for arg in sys.argv[2:]:
        if arg.startswith('--ncols='):
            ncols = int(arg.split('=', 1)[1])
    compose(out_path, items, ncols)
