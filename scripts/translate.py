#!/usr/bin/env python3
"""
Auto-translate Chinese markdown posts to English.

Usage:
    # translate a single file
    python scripts/translate.py content/posts/residual/index.md

    # translate all posts that need updating
    python scripts/translate.py --all

    # dry run (print what would be translated)
    python scripts/translate.py --all --dry-run

Requirements: pip install anthropic
Set ANTHROPIC_API_KEY environment variable.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTENT_DIR = ROOT / "content"
CACHE_FILE = ROOT / "scripts" / ".translate_cache.json"


# -- Patterns to PROTECT from translation ------------------------------------
# We replace these with placeholders before translation and restore after.

# Order matters: longer patterns first to avoid partial matches
PROTECT_PATTERNS = [
    # display math \[ ... \]
    (re.compile(r'\\\[(.*?)\\\]', re.DOTALL), 'DISPLAYMATH'),
    # inline math \( ... \)
    (re.compile(r'\\\((.*?)\\\)', re.DOTALL), 'INLINEMATH'),
    # display math $$ ... $$
    (re.compile(r'\$\$(.*?)\$\$', re.DOTALL), 'DISPLAYMATH'),
    # inline math $ ... $ (careful: single $)
    (re.compile(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', re.DOTALL), 'INLINEMATH'),
    # Hugo shortcodes {{< ... >}} and {{% ... %}}
    (re.compile(r'\{\{[<%#].*?[>%]#?\}\}', re.DOTALL), 'SHORTCODE'),
    # HTML tags (inline and block)
    (re.compile(r'<[^>]+>', re.DOTALL), 'HTMLTAG'),
    # code fences ``` ... ```
    (re.compile(r'```.*?```', re.DOTALL), 'CODEFENCE'),
    # inline code `...`
    (re.compile(r'`[^`]+`'), 'INLINECODE'),
    # Hugo frontmatter (will be handled separately)
    # URLs
    (re.compile(r'https?://[^\s\)]+'), 'URL'),
    # image references ![alt](src)
    (re.compile(r'!\[.*?\]\(.*?\)'), 'IMAGE'),
]


def protect_content(text: str) -> tuple[str, dict[str, str]]:
    """Replace protected patterns with placeholders, return (clean_text, mapping)."""
    mapping = {}
    counter = [0]

    def replacer(m, tag):
        key = f"__{tag}_{counter[0]}__"
        counter[0] += 1
        mapping[key] = m.group(0)
        return key

    for pattern, tag in PROTECT_PATTERNS:
        text = pattern.sub(lambda m: replacer(m, tag), text)

    return text, mapping


def restore_content(text: str, mapping: dict[str, str]) -> str:
    """Restore protected content from placeholders."""
    for key, value in mapping.items():
        text = text.replace(key, value)
    return text


# -- Translation -------------------------------------------------------------

SYSTEM_PROMPT = """You are a technical translator specializing in AI/ML content.

Translate the following Chinese markdown to English. STRICT rules:
1. Translate ONLY Chinese text. Leave English text untouched.
2. Placeholders like __DISPLAYMATH_0__, __INLINEMATH_1__, __SHORTCODE_2__,
   __HTMLTAG_3__, __CODEFENCE_4__, __URL_5__, __IMAGE_6__, __INLINECODE_7__
   must appear EXACTLY as-is in the output — do not modify them at all.
3. Preserve ALL markdown formatting (headings, lists, bold, italic, tables, blockquotes).
4. Preserve the exact structure (line breaks, blank lines).
5. Use natural, idiomatic English appropriate for a technical blog.
6. Do NOT add any explanation or commentary — output ONLY the translated markdown."""


def translate_text(text: str, api_key: str) -> str:
    """Translate Chinese markdown to English via Anthropic API."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    # Step 0: extract alt/caption Chinese texts before protecting
    alt_texts = {}
    counter = [0]

    def stash_alt(m):
        key = f"__ALT_{counter[0]}__"
        counter[0] += 1
        alt_texts[key] = m.group(1)
        # preserve the tag but with a placeholder alt
        return m.group(0).replace(f'"{m.group(1)}"', f'"{key}"')

    # Extract alt="..." / caption="..." Chinese text from HTML/shortcode tags
    text = re.sub(
        r'\b(?:alt|caption)="([^"]*[一-鿿][^"]*)"',
        stash_alt,
        text,
    )
    # Extract ![Chinese alt](...) from markdown images
    def stash_md_alt(m):
        key = f"__ALT_{counter[0]}__"
        counter[0] += 1
        alt_texts[key] = m.group(1)
        return f"![{key}]({m.group(2)})"

    text = re.sub(
        r'!\[([^\]]*[一-鿿][^\]]*)\]\(([^\)]+)\)',
        stash_md_alt,
        text,
    )

    # Translate alt/caption texts in batch
    if alt_texts:
        batch = "\n---\n".join(f"{k}: {v}" for k, v in alt_texts.items())
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system="Translate each Chinese alt/caption text to English. Output format: KEY: translated text. Keep one KEY: value per line.",
            messages=[{"role": "user", "content": batch}],
        )
        batch_result = "".join(b.text for b in resp.content if b.type == "text")
        for line in batch_result.strip().split("\n"):
            line = line.strip()
            if ":" in line:
                k, v = line.split(":", 1)
                k, v = k.strip(), v.strip()
                if k in alt_texts:
                    alt_texts[k] = v

    protected, mapping = protect_content(text)

    # chunk if too long (>50k chars to be safe)
    if len(protected) < 50000:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": protected}],
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        result = "".join(text_blocks)
        result = restore_content(result, mapping)
        # restore translated alts
        for k, v in alt_texts.items():
            result = result.replace(k, v)
        return result
    else:
        # split by H2 headings
        chunks = re.split(r'(?=^## )', protected, flags=re.MULTILINE)
        results = []
        for chunk in chunks:
            if not chunk.strip():
                results.append(chunk)
                continue
            if len(chunk) < 100:
                results.append(chunk)
                continue
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": chunk}],
            )
            chunk_text = "".join(b.text for b in response.content if b.type == "text")
            results.append(chunk_text)
            time.sleep(0.5)  # rate limit
        result = "\n".join(results)
        result = restore_content(result, mapping)
        for k, v in alt_texts.items():
            result = result.replace(k, v)
        return result


# -- File handling -----------------------------------------------------------

def get_hash(filepath: Path) -> str:
    """Get SHA256 hash of file content."""
    return hashlib.sha256(filepath.read_bytes()).hexdigest()


def load_cache() -> dict:
    """Load translation cache (source_hash -> en_hash)."""
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def save_cache(cache: dict):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


def needs_translation(zh_path: Path, en_path: Path, cache: dict) -> bool:
    """Check if English translation needs updating."""
    if not en_path.exists():
        return True
    zh_hash = get_hash(zh_path)
    rel = str(zh_path.relative_to(ROOT))
    cached = cache.get(rel, {})
    if not cached:
        # en exists but no cache entry — assume previously translated,
        # seed cache so we don't re-translate until zh changes
        cache[rel] = {"zh_hash": zh_hash}
        return False
    return cached.get("zh_hash") != zh_hash


def process_file(zh_path: Path, api_key: str, dry_run: bool, cache: dict):
    """Translate a single Chinese markdown file."""
    en_path = zh_path.with_name(zh_path.stem + ".en.md")

    # re-read in case stem handling was wrong (e.g., _index.md → _index.en.md)
    if zh_path.name == "_index.md":
        en_path = zh_path.with_name("_index.en.md")
    elif zh_path.name == "index.md":
        en_path = zh_path.with_name("index.en.md")

    if not needs_translation(zh_path, en_path, cache):
        return

    print(f"{'[DRY RUN] ' if dry_run else ''}Translating: {zh_path.relative_to(ROOT)}")

    if dry_run:
        return

    content = zh_path.read_text(encoding="utf-8")

    # split frontmatter and body
    parts = content.split("---", 2)
    if len(parts) >= 3 and content.strip().startswith("---"):
        fm = parts[1]
        body = parts[2]

        # translate frontmatter fields selectively
        fm_en = fm
        for key in ["title", "description", "summary"]:
            fm_en = re.sub(
                rf'({key}:\s*")([^"]*?)(")',
                lambda m, k=key: m.group(1)
                + translate_text(m.group(2), api_key).replace('"', '\\"')
                + m.group(3),
                fm_en,
            )
        # translate series names
        fm_en = re.sub(
            r'(main:\s*")([^"]*?)(")',
            lambda m: m.group(1)
            + translate_text(m.group(2), api_key).replace('"', '\\"')
            + m.group(3),
            fm_en,
        )
        fm_en = re.sub(
            r'(subseries:\s*")([^"]*?)(")',
            lambda m: m.group(1)
            + translate_text(m.group(2), api_key).replace('"', '\\"')
            + m.group(3),
            fm_en,
        )

        # translate body
        body_en = translate_text(body, api_key)

        body_en = body_en.lstrip("\n")
        output = f"---{fm_en}---\n\n{body_en}"
    else:
        output = translate_text(content, api_key)

    en_path.write_text(output, encoding="utf-8")

    # update cache
    zh_hash = get_hash(zh_path)
    en_hash = get_hash(en_path)
    cache[str(zh_path.relative_to(ROOT))] = {"zh_hash": zh_hash, "en_hash": en_hash}


def find_files(target_dir: Path) -> list[Path]:
    """Find Chinese index.md / _index.md files in directory tree."""
    files = []
    for md in target_dir.rglob("*.md"):
        if md.name.endswith(".en.md"):
            continue
        # skip if it looks like already English (heuristic)
        content = md.read_text(encoding="utf-8")
        if not re.search(r'[一-鿿]', content):
            continue
        files.append(md)
    return files


def main():
    parser = argparse.ArgumentParser(description="Auto-translate Chinese posts to English")
    parser.add_argument("file", nargs="?", help="Single file to translate")
    parser.add_argument("--all", action="store_true", help="Translate all files needing update")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be translated")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        print("Error: Set ANTHROPIC_API_KEY environment variable or pass --api-key")
        sys.exit(1)

    cache = load_cache()

    if args.file:
        files = [Path(args.file).resolve()]
    elif args.all:
        files = find_files(CONTENT_DIR)
    else:
        print("Specify a file or use --all")
        sys.exit(1)

    for f in files:
        process_file(f, api_key, args.dry_run, cache)

    if not args.dry_run:
        save_cache(cache)


if __name__ == "__main__":
    main()
