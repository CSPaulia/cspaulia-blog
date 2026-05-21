"""One-off: fix relative image paths in English markdown files to absolute paths."""
import re, glob, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for md in glob.glob(os.path.join(ROOT, "content", "**", "*.en.md"), recursive=True):
    rel = os.path.relpath(md, ROOT).replace("\\", "/")
    m = re.match(r"content/((?:posts|publications)/[^/]+)/[^/]+\.en\.md", rel)
    if not m:
        continue
    base = "/" + m.group(1)

    with open(md, encoding="utf-8") as f:
        content = f.read()

    original = content

    # <img src="..." /> and href="..."
    content = re.sub(
        r'(src|href)="(?!https?://|/|#)([^"]+\.(?:png|jpg|jpeg|gif|svg|webp|avif|pdf))"',
        rf'\1="{base}/\2"',
        content,
    )
    # Markdown ![...](...)
    content = re.sub(
        r'!\[([^\]]*)\]\((?!https?://|/|#)([^\)]+\.(?:png|jpg|jpeg|gif|svg|webp|avif|pdf))\)',
        rf'![\1]({base}/\2)',
        content,
    )

    if content != original:
        with open(md, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed: {rel}")
    else:
        print(f"Skip:  {rel}")
