# CSPaulia’s blog

Welcome to CSPaulia’s blog.

## Scripts

### Auto-translate posts to English

```bash
pip install anthropic
set ANTHROPIC_API_KEY=sk-ant-...
python scripts/translate.py --all          # translate all new/modified posts
python scripts/translate.py --all --dry-run # preview without making changes
python scripts/translate.py content/posts/foo/index.md  # translate a single file
```

Checks SHA256 hashes — only re-translates when the Chinese source has changed. Math, code, shortcodes, and HTML are preserved untouched.