"""Probes for the 1908-register confound in the 哀弦篇 analysis.

Experiment A (runs immediately): hold out all of Zhou Zuoren's 1907--1911
wenyan prefaces from training and verify that the model still attributes
them to Zhou Zuoren. This tests the *period* dimension of the confound.

Experiment B (requires text): predict 《论文章之意义暨其使命因及中国近时论文之失》
(《河南》第4、5期, 1908, signed 独应, standardly attributed to Zhou Zuoren and
claimed by him) as a fully held-out probe. This tests the *genre/register*
dimension: a long 1908 critical essay by Zhou Zuoren. Prepare the text as
probe/lunwenzhang.txt -- simplified Chinese, UTF-8, body text only (no title
or editorial notes); mask direct quotations character-by-character with '☒',
following the main corpus convention. A clean source is 《周作人集外文》
(陈子善、张铁荣编) or 《周作人散文全集》第1卷, 87--115.

Interpretation: if the probe comes back solidly Zhou Zuoren (p(LX) low, as the
prefaces and the held-out 童话 essays do), the Lu Xun-leaning signal in 哀弦篇
cannot be dismissed as a register artifact. If it comes back Lu Xun-leaning,
the reference corpus does not support register-level claims about 哀弦篇.

Run from the repository root: python -m probe.probe
"""

__license__ = "0BSD"

import os
import re
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import count_frequency, load_corpus  # noqa: E402

# the 31 features from recursive feature elimination (see run.py)
# fmt: off
NGRAMS = ['诚', '于', '乃', '光', '原', '各', '必', '惟', '不', '随', '本',
          '全', '但', '徒', '别', '是', '为', '何', '多', '夫', '则', '之',
          '焉', '皆', '而', '矣', '及', '自然', '进而', '足以', '于是']
# fmt: on
PROBE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lunwenzhang.txt")


def featurize(text: str, length: int) -> list:
    return [c / length for c in count_frequency(NGRAMS, text)]


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2", C=1.0, solver="lbfgs", tol=1e-9, max_iter=int(1e9)
                ),
            ),
        ]
    )


def clean_length(text: str) -> int:
    return len(re.sub(r"\s+", "", re.sub(r"\[([A-Za-z ]+)\]", "", text)))


def cjk_length(text: str) -> int:
    """Alternative denominator: CJK characters plus mask tokens only."""
    t = re.sub(r"\s+", "", text)
    return sum(1 for c in t if "\u4e00" <= c <= "\u9fff") + t.count("☒")


def chunk_paragraphs(text: str, target: int = 2000) -> list:
    chunks, buf = [], ""
    for para in [p for p in text.split("\n") if p.strip()]:
        buf += para + "\n"
        if clean_length(buf) >= target:
            chunks.append(buf)
            buf = ""
    if clean_length(buf) > 200:
        chunks.append(buf)
    elif chunks and buf:
        chunks[-1] += buf
    return chunks


def experiment_a(train: list) -> None:
    print("== Experiment A: 1907--1911 ZZR prefaces held out ==")
    prefaces = [
        d
        for d in train
        if d["author"] == "zzr" and "序" in d["title"] and "童话" not in d["title"]
    ]
    rest = [d for d in train if d not in prefaces]
    model = make_model()
    model.fit(
        [featurize(d["text"], d["text_length"]) for d in rest],
        [d["label"] for d in rest],
    )
    for d in prefaces:
        p = model.predict_proba([featurize(d["text"], d["text_length"])])[0, 1]
        print(f"  {d['title']:22s} len={d['text_length']:5d}  p(LX)={p:.3f}")
    merged = "".join(d["text"] for d in prefaces)
    merged_len = sum(d["text_length"] for d in prefaces)
    p = model.predict_proba([featurize(merged, merged_len)])[0, 1]
    print(f"  {'merged prefaces':22s} len={merged_len:5d}  p(LX)={p:.3f}")
    print("  (values near 0 = still recognized as Zhou Zuoren)\n")


def experiment_b(train: list) -> None:
    print("== Experiment B: 《论文章之意义》 held-out probe ==")
    if not os.path.exists(PROBE_PATH):
        print(f"  Text not found at {PROBE_PATH}.")
        print("  Ask Ogawa for the 《周作人集外文》 plaintext, prepare it per the")
        print("  module docstring, then rerun. Nothing else to do.\n")
        return
    text = open(PROBE_PATH).read()
    model = make_model()
    model.fit(
        [featurize(d["text"], d["text_length"]) for d in train],
        [d["label"] for d in train],
    )
    # cjk-denominator variant needs a consistently normalized model
    model2 = make_model()
    model2.fit(
        [featurize(d["text"], cjk_length(d["text"])) for d in train],
        [d["label"] for d in train],
    )
    # stylome density reference from the training split
    dens_ref = [
        sum(count_frequency(NGRAMS, d["text"])) / d["text_length"] for d in train
    ]
    mu, sd = float(np.mean(dens_ref)), float(np.std(dens_ref, ddof=1))
    docs = [("whole essay", text)] + [
        (f"chunk {i + 1}", c) for i, c in enumerate(chunk_paragraphs(text))
    ]
    print(f"  {'doc':12s} {'len':>6s} {'p(LX)':>7s} {'p(LX,cjk)':>10s} {'dens z':>7s}")
    for name, doc in docs:
        length_1, length_2 = clean_length(doc), cjk_length(doc)
        p1 = model.predict_proba([featurize(doc, length_1)])[0, 1]
        p2 = model2.predict_proba([featurize(doc, length_2)])[0, 1]
        z = (sum(count_frequency(NGRAMS, doc)) / length_1 - mu) / sd
        print(f"  {name:12s} {length_1:6d} {p1:7.3f} {p2:10.3f} {z:+7.2f}")
    print("  (p(LX) near 0 = attributed to Zhou Zuoren; |z|>1.96 = density atypical)\n")


if __name__ == "__main__":
    train, _, _ = load_corpus()
    experiment_a(train)
    experiment_b(train)
