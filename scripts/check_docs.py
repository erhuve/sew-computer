from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    'README.md', 'AGENTS.md', 'docs/product/vision.md',
    'docs/plans/software-prototype.md', 'docs/design/project-contract.md',
    'docs/design/editor-ux.md', 'docs/design/engine-ai.md',
    'docs/design/backend-security.md', 'docs/design/tech-pack-handoff.md',
    'docs/research/design2garmentcode-evidence.md',
    'docs/reviews/planning-adversarial.md',
]


def anchors(text):
    result = set()
    counts = {}
    for title in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', text, re.M):
        slug = re.sub(r'[^\w\s-]', '', title.lower()).replace(' ', '-')
        number = counts.get(slug, 0)
        counts[slug] = number + 1
        result.add(f'{slug}-{number}' if number else slug)
    return result


errors = []
for name in REQUIRED:
    if not (ROOT / name).is_file():
        errors.append(f'Missing required document: {name}')
paths = sorted(set(ROOT.glob('*.md')) | set((ROOT / 'docs').rglob('*.md')))
for source in paths:
    text = source.read_text()
    label = source.relative_to(ROOT)
    for number, line in enumerate(text.splitlines(), 1):
        if line.rstrip() != line:
            errors.append(f'{label}:{number}: trailing whitespace')
    for url in re.findall(r'\[[^\]]*\]\(([^\s)]+)\)', text):
        parts = urlsplit(url)
        if parts.scheme or parts.netloc:
            continue
        target = (source.parent / unquote(parts.path)).resolve() if parts.path else source
        if not target.is_relative_to(ROOT):
            errors.append(f'{label}: link escapes repository: {url}')
            continue
        if not target.exists():
            errors.append(f'{label}: missing local link: {url}')
        elif parts.fragment and target.suffix == '.md':
            if unquote(parts.fragment) not in anchors(target.read_text()):
                errors.append(f'{label}: missing heading anchor: {url}')
    if '/home/workspace/' in text or '/home/.z/' in text:
        errors.append(f'{label}: private workspace path in public documentation')

if errors:
    print('\n'.join(errors), file=sys.stderr)
    raise SystemExit(1)
print(f'Validated {len(paths)} Markdown documents: required files, local links/anchors and whitespace.')
