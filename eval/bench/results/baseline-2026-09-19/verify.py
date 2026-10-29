"""Offline verification of this fixed historical artifact bundle (stdlib only)."""
import gzip
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def verify(root):
    hashes = json.loads((root / 'SHA256SUMS.json').read_text())
    for name, expected in hashes.items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
                f'Checksum mismatch: {name}')
    audit = json.loads((root / 'audit.json').read_text())
    for dataset, expected in audit['runs'].items():
        directory = root / dataset
        rows = read_rows(directory / 'questions.jsonl.gz')
        calls = read_rows(directory / 'calls.jsonl.gz')
        summary = json.loads((directory / 'summary.json').read_text())
        manifest = json.loads((directory / 'manifest.json').read_text())
        require(manifest == summary['manifest'], f'{dataset}: manifest mismatch')
        require(manifest['dataset_sha256'] == expected['dataset_sha256'],
                f'{dataset}: dataset hash mismatch')
        require(len(rows) == expected['questions'] == len({r['question_id'] for r in rows}),
                f'{dataset}: missing or duplicate questions')
        primary, abstention, contexts = [], [], []
        for row in rows:
            for phase in ('store', 'retrieve', 'answer', 'judge'):
                require(row[phase]['status'] == 'ok', f'{dataset}: incomplete {phase}')
            result = row['judge']['result']
            if dataset == 'longmemeval' or row['abstention']:
                label = result['text'].strip().lower().rstrip('.')
                require(label in ('yes', 'no'), 'Invalid yes/no verdict')
                correct = label == 'yes'
            else:
                label = json.loads(result['text'])['label']
                require(label in ('CORRECT', 'WRONG'), 'Invalid J-score verdict')
                correct = label == 'CORRECT'
            require(type(result['correct']) is bool and result['correct'] == correct,
                    f'{dataset}: inconsistent judgment')
            if dataset == 'longmemeval' or not row['abstention']:
                primary.append(correct)
            if row['abstention']:
                abstention.append(correct)
            nodes = row['retrieve']['result']['ranked']
            require(len(nodes) <= 30, 'Retrieval budget exceeded')
            context = '\n\n'.join(
                f"[{n['node'].get('created', 'unknown date')}] "
                f"{n['node'].get('title', '')}\n{n['node'].get('content', '')}"
                for n in nodes)
            contexts.append(len(context))
            for phase in ('answer', 'judge'):
                matching = [c for c in calls if c['item_id'] == row['question_id']
                            and c['phase'] == phase and not c.get('error')]
                require(len(matching) == 1, f'{dataset}: ambiguous call ledger')
                require(matching[0]['text'] == row[phase]['result']['text'],
                        f'{dataset}: ledger text mismatch')
        require(len(primary) == expected['scored'] == summary['overall']['scored'],
                f'{dataset}: scored denominator mismatch')
        require(sum(primary) == expected['correct'], f'{dataset}: numerator mismatch')
        require(sum(primary) / len(primary) == summary['overall']['accuracy'],
                f'{dataset}: score mismatch')
        require(len(abstention) == expected['abstention_questions'] and
                sum(abstention) == expected['abstention_correct'], 'Abstention mismatch')
        require(sum(contexts) / len(contexts) == expected['mean_answer_context_chars'],
                'Context accounting mismatch')
        require(all(c['provider'] == 'codex' and c['usd'] == 0 and not c.get('error')
                    for c in calls), 'Unexpected provider, spend or call error')
        print(f'{dataset}: {sum(primary)}/{len(primary)} = '
              f'{sum(primary)/len(primary):.4%}; '
              f'abstention {sum(abstention)}/{len(abstention)}; all checks passed')


if __name__ == '__main__':
    verify(Path(__file__).resolve().parent)
