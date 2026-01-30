---
title: "Tokenization"
date: 2025-07-17T10:20:03+08:00
# weight: 1
# aliases: ["/first"]
series:
    main: "Large Language Model"
    subseries: "Tokenization"
categories: ["Large Language Model", "NLP"]
tags: ["Tokenization", "LLM"]
author: "CSPaulia"
# author: ["Me", "You"] # multiple authors
showToc: true
TocOpen: true # show table of contents
draft: false
hidemeta: false
comments: false
description: "Tokenization in LLMs"
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
    alt: "tokenization" # alt text
    caption: "Tokenization" # display caption under cover
    relative: false # when using page bundles set this to true
    hidden: true # only hide on current single page
    hiddenInList: false # hide on list pages and home
editPost:
    URL: "https://cspaulia.github.io/cspaulia-blog/content/"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

👉 Try it online: [Tokenization Visualization Tool](https://tiktokenizer.vercel.app)

---

We represent raw text as a Unicode string.

```python
string = "Hello, 🌍! 你好!"
```

Language models operate on a sequence of tokens (usually represented by integer IDs) and model a probability distribution over them.

```python
indices = [15496, 11, 995, 0]
```

We need:

- ✅ A method to **encode a string into tokens**
- ✅ A method to **decode tokens back into a string**

```python
class Tokenizer:
    def encode(self, string: str) -> list[int]:
        ...
    def decode(self, indices: list[int]) -> str:
        ...
```

- `vocab_size`: the vocabulary size, i.e., the total number of possible token IDs.

---

## 1. Character-based tokenization

### 1.1. Unicode overview

- A universal standard for character encoding (around 150,000 characters)
- `ord(char)`: get the decimal code point of a character

```python
ord("h")     # 104
ord("😊")    # 128522
```

### 1.2. Encoding and decoding

```python
class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points."""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))
    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))
```

**Example**:

```python
tokenizer = CharacterTokenizer()
string = "Hello, 🌍! 你好!"  # @inspect string
indices = tokenizer.encode(string)  # @inspect indices
reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
```

**Output**:

```text
string = "Hello, 🌍! 你好!"
indices = [72, 101, 108, 108, 111, 44, 32, 127757, 33, 32, 20320, 22909, 33]
reconstructed_string = "Hello, 🌍! 你好!"
```

### 1.3. Limitations

- Issue 1: the vocabulary can become very large.
- Issue 2: many characters occur very rarely (e.g., 🌍), which is inefficient for vocabulary usage.

    ```python
    def get_compression_ratio(string: str, indices: list[int]) -> float:
        """Return the number of UTF-8 bytes per token for a tokenized string."""
        num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
        num_tokens = len(indices)                       # @inspect num_tokens
        return num_bytes / num_tokens

    vocabulary_size = max(indices) + 1  # This is a lower bound @inspect vocabulary_size
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio
    ```

    **Output**:

    ```text
    vocabulary_size = 127758
    num_bytes = 20
    num_tokens = 13
    compression_ratio = 1.5384615384615385
    ```

---

## 2. Byte-based tokenization

- A Unicode string can be represented as a sequence of bytes. Each byte (8 bits) is a number from 0 to 255.
- The most common Unicode encoding is UTF-8.

    **Input**:

    ```python
    bytes("a", encoding="utf-8")
    bytes("🌍", encoding="utf-8")
    ```

    **Output**:

    ```text
    b"a" # one byte
    b"\xf0\x9f\x8c\x8d" # multiple bytes
    ```

### 2.1. Encoding and decoding

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

**Example**:

```python
tokenizer = ByteTokenizer()
string = "Hello, 🌍! 你好!"  # @inspect string
indices = tokenizer.encode(string)  # @inspect indices
reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
```

**Output**:

```text
string = "Hello, 🌍! 你好!"
string_bytes = "b'Hello, \\xf0\\x9f\\x8c\\x8d!\\xe4\\xbd\\xa0\\xe5\\xa5\\xbd'"
indices = [72, 101, 108, 108, 111, 44, 32, 240, 159, 140, 141, 33, 32, 228, 189, 160, 229, 165, 189, 33]
reconstructed_string = "Hello, 🌍! 你好!"
```

### 2.2. Limitations

- Issue 1: although the vocabulary is tiny (only 256), sequences become much longer. In Transformers, compute scales **quadratically** with sequence length.

    ```python
    vocabulary_size = 256  # This is a lower bound @inspect vocabulary_size
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio
    ```

    **Output**:

    ```text
    num_bytes = 20
    num_tokens = 20
    compression_ratio = 1.0
    ```

---

## 3. Word-based tokenization

Split a string using a traditional NLP-style word segmentation. **Input**:

```python
string = "I'll say supercalifragilisticexpialidocious!"
segments = regex.findall(r"\w+|.", string)  # @inspect segments
```

**Output**:

```text
segments = ["I", "ll", "say", "supercalifragilisticexpialidocious", "!"]
```

### 3.1. Encoding and decoding

- To turn this into a `Tokenizer`, we need to map each segment to an integer ID.
- Build a mapping from each segment to an integer.

### 3.2. Limitations

- The number of words is enormous.
- Many words are rare, so the model learns very little from them.
- It does not naturally provide a fixed-size vocabulary.

---

## 4. Byte Pair Encoding (BPE)

**Core idea**: train a `tokenizer` on raw text and automatically build a vocabulary.

**Motivation**: represent frequent character/byte sequences with a single token, and represent rare sequences with multiple tokens.

**High-level algorithm**: treat each **byte** as a token first, then repeatedly merge the most frequent adjacent token pair into a new token.

### 4.1. Merge procedure

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

### 4.2. A BPE tokenizer

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

### 4.3. Training BPE

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

**Example**:

```python
text = "the cat in the hat"  # @inspect string
params = train_bpe(text, num_merges=3)

string = "the quick brown fox"  # @inspect string
tokenizer = BPETokenizer(params)
indices = tokenizer.encode(string)  # @inspect indices
reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
```

What the code is doing:

1. Initialize the vocabulary with IDs 0–255 for bytes.
2. Repeatedly find the most frequent adjacent byte pair.
    - (116, 104) → 256, i.e., ('t', 'h') → 'th'
    - (256, 101) → 257, i.e., ('th', 'e') → 'the'
    - (257, 32) → 258, i.e., ('the', ' ') → 'the '
3. The vocabulary size becomes 259.
4. Encode strings using the learned merges/vocabulary.

---

<div class="zhihu-ref">
    <div class="zhihu-ref-title">References</div>
  <ol>
    <li><a href="https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture_01.json" target="_blank">stanford-cs336 lecture 1</a></li>
  </ol>
</div>