"""Knowledge Video 06 — Latent Diffusion 与 Stable Diffusion.

Usage: TTS_MODELS=<dir> [ASR_MODEL=<dir>] python ep06_latent.py <out_dir>

Toy latent diffusion, computed for real: 2000 centred-ellipse images (32x32), PCA to 16 dims as
a linear autoencoder, a 60-component GMM fitted to the latents, DDIM sampling with the exact
GMM denoiser in latent space, then decoding. Novelty is checked by nearest-neighbour distance.
Stable Diffusion facts (512x512x3 -> 64x64x4) are from the SD v1 architecture.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw

import diff2d as D
import latent as L
import vidkit as vk
from vidkit import CYAN, VIOLET, TEXT, MUTED, GREEN, F_BODY, F_SMALL, F_TITLE

DIM = 16
X = L.make_images(2000, 0)
XTEST = L.make_images(300, 7)
PCA = L.PCA(X, DIM)
Z = PCA.enc(X)
SD = Z.std()
MU, VAR, W = L.fit_gmm(Z / SD, 60, 0)
REC = PCA.dec(PCA.enc(X[:6]))
PSNR = float(10 * np.log10(1 / ((PCA.dec(PCA.enc(XTEST)) - XTEST) ** 2).mean()))
EXPLAINED = float(PCA.ev[:DIM].sum())

ZT = np.random.default_rng(3).standard_normal((300, DIM))
ZGEN, ZTRAJ = L.ddim_latent(ZT, MU, VAR, W, 50, D.abar, keep=True)
GEN = PCA.dec(ZGEN * SD)
# predicted clean latents along the trajectory, for the decoding animation (first 6 samples)
TS = np.linspace(1.0, 1e-3, 51)
PRED = [PCA.dec(L.x0_hat(ZTRAJ[i, :6], D.abar(TS[min(i, 50)]), MU, VAR, W) * SD) for i in range(ZTRAJ.shape[0])]


def nn(A):
    d = np.sqrt(((A[:, None] - X[None]) ** 2).mean(-1))
    return d.min(1), d.argmin(1)


GEN_NN, GEN_IDX = nn(GEN)
TEST_NN, _ = nn(XTEST)
print(f"explained {EXPLAINED:.3f}  test PSNR {PSNR:.1f} dB  gen NN median {np.median(GEN_NN):.4f}  test NN median {np.median(TEST_NN):.4f}")
assert 0.94 < EXPLAINED and 26 < PSNR
assert np.median(GEN_NN) > 1.5 * np.median(TEST_NN)  # generated images are not copies

# ---------------------------------------------------------------------------
# Narration
# ---------------------------------------------------------------------------
SCENES = [
    ("title", [
        "前几期的例子都在二维平面上。但真实的图片动辄几十万个像素，如果直接在像素上做扩散，每一步都要处理全部像素，计算量非常大。",
        "这一期讲潜空间扩散，也就是稳定扩散背后的核心思想。",
    ]),
    ("pixels", [
        "以稳定扩散为例，一张五百一十二乘五百一十二的彩色图片，有七十八万多个数值。",
        "但图片里有大量冗余，相邻的像素高度相关，很多细节人眼也并不敏感。",
        "所以思路是，先把图片压缩到一个小得多的潜空间，在潜空间里做扩散，最后再解码回图片。",
    ]),
    ("ae", [
        "压缩用的是一个自编码器，编码器把图片变成潜变量，解码器再把它还原。",
        "稳定扩散的自编码器把长和宽各缩小八倍，得到六十四乘六十四乘四的潜变量，数值的个数少了四十八倍。",
        "我们用一个小例子来演示。两千张三十二乘三十二的椭圆图片，用主成分分析作为最简单的线性自编码器，压缩到十六维，也就是缩小了六十四倍。",
        "重建的结果和原图已经很接近，只是边缘有一些波纹，这是线性压缩的局限。",
    ]),
    ("latent", [
        "然后在这个十六维的潜空间里做扩散。这里我们用高斯混合模型来近似潜变量的分布，并用精确的去噪器采样。",
        "从随机噪声出发，在潜空间里去噪，再交给解码器，就得到了新的椭圆图片。",
        "下面一行，是训练集中和它们最接近的图片，可以看到，生成的图片并不是训练集的复制。",
        "真实的稳定扩散，用的是非线性的变分自编码器和神经网络去噪器，效果要好得多。",
    ]),
    ("text", [
        "怎样加入文字条件呢？",
        "文字先经过一个预训练的文本编码器，变成一串向量，每个词元对应一个向量。",
        "去噪网络里有交叉注意力层，用图像特征作为查询，用文字向量作为键和值，这样图像的每个位置，都能关注到和它相关的词语。",
    ]),
    ("pipeline", [
        "完整的生成流程是这样的。",
        "先把文字编码，再在潜空间里从随机噪声出发，配合无分类器引导，迭代去噪几十步，最后用解码器还原成图片。",
        "因为去噪在小得多的潜空间里进行，训练和生成都便宜得多，这也是稳定扩散能在消费级显卡上运行的重要原因。",
    ]),
    ("summary", [
        "总结一下。",
        "第一，像素空间维度很高，直接扩散代价很大。",
        "第二，自编码器把图片压缩到潜空间，扩散在潜空间里进行，最后再解码。",
        "第三，文字通过文本编码器和交叉注意力，作为条件来引导去噪。",
        "下一期，我们讲流匹配，看看另一种把噪声变成数据的方法。感谢大家的观看。",
    ]),
]

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def img(arr, size):
    a = (np.clip(arr.reshape(32, 32), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(a, "L").resize((size, size), Image.NEAREST).convert("RGB")


def box(d, xy, label, col=CYAN, sub=None, w=190, h=90):
    x, y = xy
    d.rounded_rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], 14, outline=col, width=3, fill=(12, 19, 34))
    vk.text_c(d, (x, y - (12 if sub else 0)), label, F_BODY, col)
    if sub:
        vk.text_c(d, (x, y + 24), sub, F_SMALL, MUTED)


def fpaste(c, tex, cx, cy, size=28, color="#e2e8f0"):
    vk.paste_center(c, vk.formula(tex, size, color), cx, cy)


def bars(d, x0, y0, z, w=12, h=120, col=VIOLET):
    m = np.abs(z).max() + 1e-9
    for i, v in enumerate(z):
        bh = v / m * h / 2
        d.rectangle([x0 + i * (w + 3), y0 - max(bh, 0), x0 + i * (w + 3) + w, y0 - min(bh, 0)], fill=col)
    d.line([(x0 - 4, y0), (x0 + len(z) * (w + 3), y0)], fill=(71, 85, 105), width=1)


def sc_title(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.text_c(d, (640, 190), "Knowledge Video · 06", vk.font(24, mono=True), MUTED)
    vk.text_c(d, (640, 270), "潜空间扩散", F_TITLE, CYAN)
    vk.text_c(d, (640, 340), "Latent Diffusion 与 Stable Diffusion", F_BODY, TEXT)


def sc_pixels(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 1, "像素空间太大")
    vk.text_c(d, (640, 190), "512 × 512 × 3 = 786,432", vk.font(54, mono=True), VIOLET)
    vk.text_c(d, (640, 250), "一张图片的数值个数", F_SMALL, MUTED)
    if seg >= 1:
        vk.text_c(d, (640, 330), "相邻像素高度相关 · 很多细节人眼不敏感", F_BODY, TEXT)
    if seg >= 2:
        for i, (lab, col) in enumerate([("图片", TEXT), ("压缩到潜空间", CYAN), ("在潜空间扩散", VIOLET), ("解码回图片", GREEN)]):
            x = 220 + i * 280
            box(d, (x, 450), lab, col, w=220, h=80)
            if i < 3:
                vk.arrow(d, (x + 115, 450), (x + 165, 450), MUTED, 3)


def sc_ae(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 2, "自编码器：压缩与还原")
    if seg <= 1:
        box(d, (330, 260), "编码器", CYAN, "Encoder")
        box(d, (950, 260), "解码器", GREEN, "Decoder")
        vk.text_c(d, (100, 260), "图片", F_BODY, TEXT)
        vk.text_c(d, (640, 260), "潜变量 z", F_BODY, VIOLET)
        vk.text_c(d, (1190, 260), "图片", F_BODY, TEXT)
        for a, b in [((150, 260), (230, 260)), ((430, 260), (560, 260)), ((720, 260), (850, 260)), ((1050, 260), (1140, 260))]:
            vk.arrow(d, a, b, MUTED, 3)
        if seg >= 1:
            vk.text_c(d, (640, 400), "Stable Diffusion: 512×512×3  →  64×64×4", vk.font(32), VIOLET)
            vk.text_c(d, (640, 450), "长宽各缩小 8 倍，数值少 48 倍", F_BODY, TEXT)
        return
    vk.text_c(d, (640, 130), f"玩具例子：32×32 = 1024 维  →  {DIM} 维（64 倍）", F_BODY, VIOLET)
    for i in range(6):
        x = 120 + i * 180
        c.paste(img(X[i], 120), (x, 180))
        if seg >= 3 or (seg == 2 and su > 0.5):
            c.paste(img(REC[i], 120), (x, 380))
    vk.text_c(d, (60, 240), "原图", F_SMALL, MUTED)
    if seg >= 3 or (seg == 2 and su > 0.5):
        vk.text_c(d, (60, 440), "重建", F_SMALL, MUTED)
        vk.text_c(d, (640, 540), f"保留 {EXPLAINED:.0%} 的方差 · 测试集重建 PSNR {PSNR:.1f} dB", F_SMALL, TEXT)


def sc_latent(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 3, "在潜空间里扩散")
    n = len(PRED) - 1
    k = 0 if seg == 0 else (int(n * vk.ease(min(1, su * 1.3))) if seg == 1 else n)
    for i in range(6):
        x = 120 + i * 180
        c.paste(img(PRED[k][i], 120), (x, 150))
        bars(d, x + 2, 335, ZTRAJ[min(k, ZTRAJ.shape[0] - 1), i] if k < n else ZGEN[i], w=5, h=60)
        if seg >= 2:
            c.paste(img(X[GEN_IDX[i]], 120), (x, 400))
            vk.text_c(d, (x + 60, 540), f"距离 {GEN_NN[i]:.3f}", F_SMALL, MUTED)
    vk.text_c(d, (60, 210), "解码", F_SMALL, MUTED)
    vk.text_c(d, (60, 335), "潜变量", F_SMALL, VIOLET)
    vk.text_c(d, (640, 120), f"去噪步数 {min(k, 50)}/50", F_SMALL, VIOLET)
    if seg >= 2:
        vk.text_c(d, (60, 460), "最近的\n训练图", F_SMALL, MUTED)
        vk.text_c(d, (640, 580), f"（新的测试图片与训练集的距离中位数 {np.median(TEST_NN):.3f}，生成图片为 {np.median(GEN_NN):.3f}）", F_SMALL, MUTED)


def sc_text(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 4, "文字条件：交叉注意力")
    vk.text_c(d, (200, 200), "“一只戴墨镜的猫”", F_BODY, TEXT)
    if seg >= 1:
        vk.arrow(d, (200, 230), (200, 280), MUTED, 3)
        box(d, (200, 330), "文本编码器", VIOLET, w=240)
        for j in range(6):
            d.rounded_rectangle([110 + j * 32, 420, 134 + j * 32, 520], 6, fill=(76, 58, 140))
        vk.text_c(d, (200, 545), "词元向量", F_SMALL, VIOLET)
    if seg >= 2:
        box(d, (800, 330), "交叉注意力", CYAN, w=260)
        vk.arrow(d, (310, 470), (670, 360), VIOLET, 3)
        vk.text_c(d, (470, 380), "键 K、值 V", F_SMALL, VIOLET)
        vk.arrow(d, (800, 200), (800, 280), CYAN, 3)
        vk.text_c(d, (800, 180), "图像特征 → 查询 Q", F_SMALL, CYAN)
        fpaste(c, r"$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$", 900, 520, 28)


def sc_pipeline(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 5, "完整流程")
    steps = [((150, 230), "文字", TEXT, None), ((400, 230), "文本编码器", VIOLET, None),
             ((150, 420), "随机噪声", TEXT, "64×64×4"), ((470, 420), "去噪网络", CYAN, "× 几十步 + CFG"),
             ((790, 420), "解码器", GREEN, None), ((1100, 420), "图片", TEXT, "512×512×3")]
    show = len(steps) if seg >= 1 else 0
    for i, (xy, lab, col, sub) in enumerate(steps[:show]):
        box(d, xy, lab, col, sub, w=230, h=90)
    if seg >= 1:
        vk.arrow(d, (265, 230), (285, 230), MUTED, 3)
        vk.arrow(d, (430, 275), (460, 370), VIOLET, 3)
        vk.arrow(d, (265, 420), (355, 420), MUTED, 3)
        vk.arrow(d, (585, 420), (675, 420), MUTED, 3)
        vk.arrow(d, (905, 420), (985, 420), MUTED, 3)
        vk.text_c(d, (470, 505), "（潜空间）", F_SMALL, MUTED)
    if seg >= 2:
        vk.text_c(d, (640, 570), "去噪在潜空间进行：训练和生成都便宜得多", F_BODY, GREEN)


def sc_summary(c, u, seg, su):
    d = ImageDraw.Draw(c)
    vk.header(d, 6, "总结")
    items = [
        ("像素空间", "维度很高，直接扩散代价大"),
        ("潜空间", "自编码器压缩 → 潜空间扩散 → 解码"),
        ("文字条件", "文本编码器 + 交叉注意力"),
    ]
    for i, (k, v) in enumerate(items):
        if seg >= i + 1:
            y = 140 + i * 105
            d.rounded_rectangle([140, y, 1140, y + 82], 14, outline=(30, 41, 59), fill=(12, 19, 34))
            d.text((180, y + 41), f"0{i + 1}", font=vk.font(30, mono=True), fill=CYAN, anchor="lm")
            d.text((260, y + 41), k, font=F_BODY, fill=TEXT, anchor="lm")
            d.text((450, y + 41), v, font=F_BODY, fill=MUTED, anchor="lm")
    if seg >= 4:
        vk.text_c(d, (640, 520), "下一期：Flow Matching 与 Rectified Flow", F_SMALL, VIOLET)


RENDER = {"title": sc_title, "pixels": sc_pixels, "ae": sc_ae, "latent": sc_latent, "text": sc_text,
          "pipeline": sc_pipeline, "summary": sc_summary}

if __name__ == "__main__":
    vk.render(SCENES, RENDER, sys.argv[1], "latent-diffusion")
