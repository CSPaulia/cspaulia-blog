# Agent Rules

This repository is a personal Hugo blog. When editing posts, follow the author's existing writing style and keep changes focused.

## Writing Style

- Prefer concise, clear Chinese explanations.
- Keep paragraphs short and easy to scan.
- Do not over-summarize the author's notes. When整理笔记, mainly preserve and list the user's original notes, only removing unclear or overly minor details.
- When adding explanations, use a simple logical flow: first the intuition, then the mechanism, then the limitation or takeaway.
- Avoid large decorative rewrites unless explicitly requested.

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
- Run `hugo --enableGitInfo=false` after meaningful Markdown or asset changes when feasible.
- For references, follow the blog's existing reference style, often:

```html
<div class="zhihu-ref">
  <div class="zhihu-ref-title">参考文献</div>
  <ol>
    <li><a href="..." target="_blank">Title</a></li>
  </ol>
</div>
```

## Image Assets

- Keep images in the same post folder when they belong to a post.
- Remove unused images only after checking that the post no longer references them.
- Do not delete unrelated assets outside the requested post folder.

## Git And Local Work

- Do not revert user changes unless explicitly requested.
- The worktree may be dirty; ignore unrelated changes.
- Use `apply_patch` for manual file edits.
