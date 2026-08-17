# Agent Rules

This repository is a personal Hugo blog. When editing posts, follow the author's existing writing style and keep changes focused.

## Writing Style

- Prefer concise, clear Chinese explanations.
- Keep paragraphs short and easy to scan.
- Do not over-summarize the author's notes. When整理笔记, mainly preserve and list the user's original notes, only removing unclear or overly minor details.
- When adding explanations, use a simple logical flow: first the intuition, then the mechanism, then the limitation or takeaway.
- Avoid large decorative rewrites unless explicitly requested.
- In article正文, avoid source-oriented phrases such as “PPT 中”“课件中” or “第几页提到”. State the fact or conclusion directly and objectively; keep source attribution in figure captions or references when needed.

## Blog Structure And Headings

- This is primarily a Chinese blog. Write headings in Chinese; when a heading introduces a specific English technical term, use `中文（English）` instead of an English-only heading.
- Make every heading understandable on its own in the table of contents. State the subject being discussed instead of relying on the preceding paragraph for context.
- Avoid vague headings such as “完整曲线”“为什么会出现幂律” or “相关内容”. Name the object explicitly, for example “数据—性能曲线的三个区间” or “数据—性能曲线为什么会呈现幂律”.
- Let headings describe the actual relationship, boundary, or conclusion of the section. Do not use an author name alone when the reader is really looking for a concept or result.
- Keep the hierarchy logical. Distinguish a basic law, its theoretical intuition, its limitations, and its applications rather than presenting them as unrelated topics.
- When writing incrementally from slides, preserve the slide order and do not create large future sections before the user asks for them.
- Put the main conclusion in the visible text first. Place long derivations, supporting experiments, secondary paper examples, or implementation details in a `<details>` block when they would interrupt the main argument.

## English Terms

- For an English technical term that appears for the first time, write it as `中文（English Full Name，ABBR）` when an abbreviation exists.
- If there is no common abbreviation, write it as `中文（English Full Name）`.
- After the first definition, abbreviations or English terms may be used naturally.
- Put a space between Chinese and English words when they appear side by side.

## PDF And Slide Processing

- When the user asks to read PDF pages, extract both text and useful images.
- Translate and organize the PDF text into Chinese before adding it to the post.
- Do not paste whole PPT pages directly into the blog unless the user explicitly asks for that.
- Prefer extracting or cropping the relevant diagram, table, or figure from the slide.
- Crop images carefully. Keep the figure complete, remove irrelevant slide margins, and avoid cutting off labels, captions, or formulas.
- For screenshots from papers or slides, write meaningful figure captions based on the content, not generic captions like "related screenshot".
- If the PDF contains formulas as images, convert them into Markdown/LaTeX formulas when practical.

## Markdown And Hugo

- Be careful with bold text next to Chinese punctuation. If Markdown emphasis fails in Hugo, use `<strong>...</strong>`.
- Avoid patterns like `**中文（English） **的`; remove extra spaces or use HTML `<strong>`.
- Write inline formulas with `\(` and `\)`, not ordinary parentheses or `$...$` delimiters. Write display formulas with `\[` and `\]`, not `$$` delimiters.
- Inside display formulas, never put a bare `=` on its own source line. Hugo's Markdown parser can treat it as a Setext heading and break the formula. Keep `=` beside an expression or use `\begin{aligned}` with `&=`.
- Do not add slide-location prose such as “CS336 Lecture 9，第 14 页” to the article body. Preserve source attribution through figure captions and the references section instead.
- Clearly distinguish empirical observations, modeling assumptions, theoretical intuition, and proven theorems. Do not present an empirical Scaling Law as a universal theorem.
- Run `hugo --enableGitInfo=false` after meaningful Markdown or asset changes when feasible.
- For references, use a plain Markdown section in the style already used by `content/posts/gpus/index.md`:

```markdown
## 参考文献

[1] Title. [Online]. Available: https://...
```

## Image Assets

- Keep images in the same post folder when they belong to a post.
- Remove unused images only after checking that the post no longer references them.
- Do not delete unrelated assets outside the requested post folder.

## Git And Local Work

- Do not revert user changes unless explicitly requested.
- The worktree may be dirty; ignore unrelated changes.
- Use `apply_patch` for manual file edits.
