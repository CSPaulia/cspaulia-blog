---
title: "分词（Tokenization）"
date: 2025-07-17T10:20:03+08:00
# weight: 1
# aliases: ["/first"]
series:
    main: "大语言模型"
    subseries: "分词"
categories: ["大语言模型", "自然语言处理"]
tags: ["分词", "LLM"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "大语言模型中的分词方法：字符/字节/词/BPE"
# canonicalURL: "https://canonical.url/to/page"
disableShare: false
disableHLJS: false
hideSummary: true
searchHidden: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "tokenization_cover.jpg" # image path/url
    alt: "分词（Tokenization）" # alt text
    caption: "分词（Tokenization）" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "建议修改" # edit text
    appendFilePath: true # to append file path to Edit link
---

👉 在线体验地址：[Tokenization 可视化工具](https://tiktokenizer.vercel.app)

---

原始的文本统一表征为 Unicode 字符串

```python
string = "Hello, 🌍! 你好!"
```

语言模型会对一系列token（通常用整数索引表示）上的可能性进行建模，构成一个概率分布

```python
indices = [15496, 11, 995, 0]
```

我们需要：

- ✅ 一个方法：**将字符串编码为 token**
- ✅ 一个方法：**将 token 解码回字符串**

```python
class Tokenizer:
    def encode(self, string: str) -> list[int]:
        ...
    def decode(self, indices: list[int]) -> str:
        ...
```

- `vocab_size`: 词表大小，即可能出现的 token（整数 ID）总数。

---

## 1. Character-based tokenization

### 1.1. Unicode 概述

- 统一全球字符编码的标准（约 150,000 个字符）
- `ord(char)`：获取字符的十进制编码

```python
ord("h")     # 104
ord("😊")    # 128522
```

### 1.2. 编解码

```python
class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points."""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))
    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))
```

**示例**：

```python
tokenizer = CharacterTokenizer()
string = "Hello, 🌍! 你好!"  # @inspect string
indices = tokenizer.encode(string)  # @inspect indices
reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
```

**输出**：

```text
string = "Hello, 🌍! 你好!"
indices = [72, 101, 108, 108, 111, 44, 32, 127757, 33, 32, 20320, 22909, 33]
reconstructed_string = "Hello, 🌍! 你好!"
```

### 1.3. 存在的问题

- 问题一：这会是一个相当大的词汇表（vocabulary）
- 问题二：很多字符出现几率很低（例如🌍），对词汇表的使用并不高效

    ```python
    def get_compression_ratio(string: str, indices: list[int]) -> float:
        """Given `string` that has been tokenized into `indices`, ."""
        num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
        num_tokens = len(indices)                       # @inspect num_tokens
        return num_bytes / num_tokens

    vocabulary_size = max(indices) + 1  # This is a lower bound @inspect vocabulary_size
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio
    ```

    **输出**：

    ```text
    vocabulary_size = 127758
    num_bytes = 20
    num_tokens = 13
    compression_ratio = 1.5384615384615385
    ```

---

## 2. Byte-based tokenization

- Unicode 字符串（String）可以表示为一串字节（Byte），其中字节（即八位二进制）可以表示为0到255的数字
- 最常见的 Unicode 编码是 UTF-8

    **输入**：

    ```python
    bytes("a", encoding="utf-8")
    bytes("🌍", encoding="utf-8")
    ```

    **输出**：

    ```text
    b"a" # one byte
    b"\xf0\x9f\x8c\x8d" # multiple bytes
    ```

### 2.1. 编解码

```python
class ByteTokenizer(Tokenizer):
    """Represent a string as a sequence of bytes."""
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")  # @inspect string_bytes
        indices = list(map(int, string_bytes))  # @inspect indices
        return indices
    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)  # @inspect string_bytes
        string = string_bytes.decode("utf-8")  # @inspect string
        return string
```

**示例**：

```python
tokenizer = ByteTokenizer()
string = "Hello, 🌍! 你好!"  # @inspect string
indices = tokenizer.encode(string)  # @inspect indices
reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
```

**输出**：

```text
string = "Hello, 🌍! 你好!"
string_bytes = "b'Hello, \\xf0\\x9f\\x8c\\x8d!\\xe4\\xbd\\xa0\\xe5\\xa5\\xbd'"
indices = [72, 101, 108, 108, 111, 44, 32, 240, 159, 140, 141, 33, 32, 228, 189, 160, 229, 165, 189, 33]
reconstructed_string = "Hello, 🌍! 你好!"
```

### 2.2. 存在的问题

- 问题一：虽然词汇表很小（仅为256），但这也导致序列很长。而在 Transformer 中，计算复杂度是随着序列长度**二次增长**的

    ```python
    vocabulary_size = 256  # This is a lower bound @inspect vocabulary_size
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio
    ```

    **输出**：

    ```text
    num_bytes = 20
    num_tokens = 20
    compression_ratio = 1.0
    ```

---

## 3. Word-based tokenization

使用类似于传统NLP分词方法分离字符串，**输入**：

```python
string = "I'll say supercalifragilisticexpialidocious!"
segments = regex.findall(r"\w+|.", string)  # @inspect segments
```

**输出**：

```text
segments = ["I", "ll", "say", "supercalifragilisticexpialidocious", "!"]
```

### 3.1. 编解码

- 要将其转换为一个`tokenizer`，我们需要将这些片段映射为整数
- 构建一个从每个片段到整数的映射

### 3.2. 存在的问题

- 词的数量是非常庞大的
- 很多词很少出现，模型不会从这些词中学习到很多内容
- 它无法提供一个固定长度的词典

---

## 4. Byte Pair Encoding（BPE）

**主要思想**：在原始文本上训练`tokenizer`，自发的生成词汇表

**意图**：对于常见的字符序列，可以仅用一个token表示；对于不常见的字符序列，则用多个token表示

**算法简述**：首先将每一个**byte**看作是一个token，随后逐渐合并常出现的相邻token为一个新token

### 4.1. 算法流程

```python
def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices
```

### 4.2. BPE 编码器

```python
class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params
    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)
        return indices
    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string
```

### 4.3. 训练 BPE

```python
def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    # Start with the list of bytes of string.
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes

    for i in range(num_merges):
        # Count the number of occurrences of each pair of tokens
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            counts[(index1, index2)] += 1  # @inspect counts

        # Find the most common pair.
        pair = max(counts, key=counts.get)  # @inspect pair
        index1, index2 = pair

        # Merge that pair.
        new_index = 256 + i  # @inspect new_index
        merges[pair] = new_index  # @inspect merges
        vocab[new_index] = vocab[index1] + vocab[index2]  # @inspect vocab
        indices = merge(indices, pair, new_index)  # @inspect indices
    
    return BPETokenizerParams(vocab=vocab, merges=merges)
```

**示例**:

```python
text = "the cat in the hat"  # @inspect string
params = train_bpe(text, num_merges=3)

string = "the quick brown fox"  # @inspect string
tokenizer = BPETokenizer(params)
indices = tokenizer.encode(string)  # @inspect indices
reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
```

代码逻辑为：

1. 初始化词汇表，用0-255表征byte
2. 依次寻找最多次出现的相邻byte
   - (116, 104) --> 256 即 ('t', 'h') --> 'th'
   - (256, 101) --> 257 即 ('th', 'e') --> 'the'
   - (257, 32) --> 258 即 （'the', ' '）--> 'the '
3. 词汇表长度更新至259
4. 用新词汇表对字符串进行编码

---

## 参考文献

[1] Stanford CS336, "Lecture 1." [Online]. Available: https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture_01.json.
