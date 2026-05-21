"""Revert absolute image paths in English files back to relative."""
import re, glob, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for md in glob.glob(os.path.join(ROOT, "content", "**", "*.en.md"), recursive=True):
    rel = os.path.relpath(md, ROOT).replace("\\", "/")
    m = re.match(r"content/((?:posts|publications)/[^/]+)/[^/]+\.en\.md", rel)
    if not m:
        continue
    base = "/" + m.group(1) + "/"

    with open(md, encoding="utf-8") as f:
        content = f.read()

    original = content

    # Revert absolute paths back to relative
    content = content.replace(
        f'src="{base}', 'src="'
    ).replace(
        f'href="{base}', 'href="'
    ).replace(
        f'src="/{base}', 'src="/'
    )

    # Revert markdown images
    content = content.replace(
        f']({base}', ']('
    ).replace(
        f'](/{base}', '](/'
    )

    if content != original:
        with open(md, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Reverted: {rel}")
    else:
        print(f"Skip:    {rel}")
