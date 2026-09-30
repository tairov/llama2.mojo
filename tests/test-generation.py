#!/usr/bin/env python3
"""Check CLI generation across threads; optionally compare a reference binary.

Build llama2 first, then run:
  python3 tests/test-generation.py --models /path/to/models
  python3 tests/test-generation.py --models /path/to/models --reference /path/to/baseline
"""
import argparse
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    (1, 0, 1, 'Once upon a time'),
    (2, 0, 1, 'Once upon a time'),
    (16, 0, 1, ''),
    (32, 0, 1, 'Hello'),
    (0, 0, 1, 'Once upon a time'),
    (64, 0.8, 1, 'Once upon a time'),
    (64, 0.8, 42, 'Once upon a time'),
    (4, 0, 1, 'Once upon a time there was a little girl who lived in a forest'),
]


def generate(binary, model, threads, case):
    tokens, temperature, seed, prompt = case
    result = subprocess.run(
        [str(binary), str(model), '-n', str(tokens), '-t', str(temperature),
         '-s', str(seed), '-j', str(threads), '-i', prompt],
        cwd=ROOT, capture_output=True, text=True, check=True, timeout=60,
    )
    assert 'achieved tok/s:' in result.stdout, 'generation did not finish'
    return re.sub(r'num parallel workers:.*\n|achieved tok/s:.*', '', result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'llama2')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--models', type=Path, default=Path(os.environ.get('MODELS', str(ROOT))))
    args = parser.parse_args()
    binary = args.binary.resolve()
    reference = (args.reference or args.binary).resolve()
    model = (args.models / 'stories15M.bin').resolve()
    for case in CASES:
        expected = generate(reference, model, 1, case)
        for threads in (1, 2, 4, 8):
            assert generate(binary, model, threads, case) == expected, (case, threads)
        print(f'PASS at 1/2/4/8 workers: {case}', flush=True)


if __name__ == '__main__':
    main()
