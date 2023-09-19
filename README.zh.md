# 四手联弹

[![en](https://img.shields.io/badge/lang-en-green.svg)](https://codeberg.org/hainingwang/four_hands/src/branch/main/README.md)
[![zh](https://img.shields.io/badge/lang-zh-green.svg)](https://codeberg.org/hainingwang/four_hands/src/branch/main/README.zh.md)

此仓库包含用于复现《四手联弹：周作人与鲁迅在<哀弦篇>中的合作》研究发现的语料库和脚本。

我们发现了证据支持《哀弦篇》，传统上被归功于周作人的独立作品，不太可能由周作人独自完成。请参考我们的[论文](#citation)以获取详细信息。

<img src="assets/axp.jpg" width="25%">

## 复现

```python3.10
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m run
```

## 语料库

| 划分 | 标题        | 作者/笔名  |
|----|-----------|--------|
| 训练 | 科学史教篇     | 鲁迅     |
|    | 文化偏至论     | 鲁迅     |
|    | 《匈奴奇士录》序  | 周作人    |
|    | 《炭画》序     | 周作人    |
|    | 《红星佚史》序   | 周作人    |
|    | 《黄蔷薇》序    | 周作人    |
|    | 童话略论      | 周作人    |
|    | 童话研究      | 周作人    |
| 验证 | 说鈤        | 鲁迅     |
|    | 摩罗诗力说     | 鲁迅     |
|    | 《秋草园日记》序  | 周作人    |
|    | 乙巳日记附记一则  | 周作人    |
|    | 江南考先生之一斑  | 周作人    |
|    | 汽船之窘况及苦热  | 周作人    |
|    | 望越篇       | 周作人和鲁迅 |
| 测试 | 哀弦篇       | 独应     |

## Visualization

<img src="assets/哀弦篇_01.jpg" width="50%">

以《哀弦篇》的第一部分为例。红色调的字符表示支持鲁迅为作者的特征，而灰色字符则暗示周作人的作者身份。颜色越深，与每个特征相关的权重的绝对值越大。
值得注意的是，支持鲁迅的特征遍布整个部分。实际上，*《哀弦篇》*的第一部分预测是由鲁迅创作的，概率为0.976。

前往文件夹 
[visualization](https://codeberg.org/hainingwang/four_hands/src/branch/main/visualization)查看更多.


## 许可证

语料库已进入公共领域。所有其他材料均根据0BSD许可证授权。

## 引用

待处理


## 联系方式
- [rwxiexin@shnu.edu.cn](mailto:rwxiexin@shnu.edu.cn) 一般性问题咨询。
- [hw56@indiana.edu](mailto:hw56@indiana.edu) 复现相关问题咨询。

## 鸣谢

该这个项目得到了中国国家社会科学基金（22CTQ041）的支持。