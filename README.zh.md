# 四手联弹

[![en](https://img.shields.io/badge/lang-en-green.svg)](https://github.com/Wang-Haining/four_hands/blob/main/README.md)
[![zh](https://img.shields.io/badge/lang-zh-green.svg)](https://github.com/Wang-Haining/four_hands/blob/main/README.zh.md)

本仓库收录了论文《四手联弹：〈哀弦篇〉著者考》的语料库与复现代码。

我们发现证据表明，传统上被认为仅由周作人独立创作的《哀弦篇》，很可能是与鲁迅合作完成的作品：
- 第一节极有可能是鲁迅独立创作。
- 第二至四节与第六节展现出两人风格的深度交融，表明兄弟二人进行了密切合作。
- 第五节判别特征密度过低、超出模型验证范围，我们不对其作者归属下判断。

详细内容请参阅我们的[论文](#citation)。

<img src="assets/axp.jpg" width="30%">

## 复现

```python3.10
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m run
```

## 语料库

| 划分 | 标题       | 作者/笔名  |
|----|----------|--------|
| 训练 | 科学史教篇    | 鲁迅     |
|    | 文化偏至论    | 鲁迅     |
|    | 《匈奴奇士录》序 | 周作人    |
|    | 《炭画》序    | 周作人    |
|    | 《红星佚史》序  | 周作人    |
|    | 《黄蔷薇》序   | 周作人    |
|    | 童话略论     | 周作人    |
|    | 童话研究     | 周作人    |
| 验证 | 说鈤       | 鲁迅     |
|    | 摩罗诗力说    | 鲁迅     |
|    | 《秋草园日记》序 | 周作人    |
|    | 乙巳日记附记一则 | 周作人    |
|    | 江南考先生之一斑 | 周作人    |
|    | 汽船之窘况及苦热 | 周作人    |
| 测试 | 望越篇（已知合作，参照样本） | 周作人和鲁迅 |
|    | 哀弦篇      | 独应     |
|    | 哀弦篇, 第一节 | 独应 |
|    | 哀弦篇, 第二节 | 独应 |
|    | 哀弦篇, 第三节 | 独应 |
|    | 哀弦篇, 第四节 | 独应 |
|    | 哀弦篇, 第五节 | 独应 |
|    | 哀弦篇, 第六节 | 独应 |

## Visualization

<img src="assets/哀弦篇_01.jpg" width="70%">

以《哀弦篇》的第一部分为例。红色调的字符表示支持鲁迅为作者的特征，而灰色字符则暗示周作人的作者身份。颜色越深，与每个特征相关的权重的绝对值越大。
值得注意的是，支持鲁迅的特征遍布整个部分。实际上，*《哀弦篇》*的第一部分预测是由鲁迅创作的，概率为0.976。

前往文件夹
[visualization](https://github.com/Wang-Haining/four_hands/tree/main/visualization)查看更多.


## 许可证

语料库已进入公共领域。所有其他材料均根据0BSD许可证授权。

## 引用

待处理


## 联系方式
- [rwxiexin@shnu.edu.cn](mailto:rwxiexin@shnu.edu.cn) 一般性问题咨询。
- [hw56@iu.edu](mailto:hw56@iu.edu) 复现相关问题咨询。

## 鸣谢

本项目得到了中国国家社会科学基金（22CTQ041）的支持。
