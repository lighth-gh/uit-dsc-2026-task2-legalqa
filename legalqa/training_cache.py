"""Read training retrieval one record at a time, including legacy JSON caches."""
import os
from pathlib import Path

from .io import digest, source_hash
from .models import model_lock


def checked_events(stream):
    import ijson
    if stream.read(3) != b'\xef\xbb\xbf':
        stream.seek(0)
    # Validate duplicates even in objects that the consumer skips. Keep keys,
    # not context strings; memory grows with IDs rather than corpus text.
    stack = []
    try:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if event == 'start_map':
                stack.append(set())
            elif event == 'map_key':
                if value in stack[-1]:
                    raise ValueError(f'Duplicate JSON key: {value}')
                stack[-1].add(value)
            elif event == 'end_map':
                stack.pop()
            yield prefix, event, value
    except ijson.JSONError as error:
        raise ValueError('Invalid or incomplete retrieval JSON') from error


def stream_identity(stream):
    import ijson
    identities = list(ijson.items(checked_events(stream), 'identity'))
    if len(identities) != 1 or not isinstance(identities[0], dict):
        raise ValueError('Retrieval requires exactly one identity object')
    return identities[0]


def stream_records(stream):
    import ijson
    yield from ijson.kvitems(checked_events(stream), 'records')


def training_retrieval(path, questions, c, root):
    """Validate identity before tokenization, and IDs/text while streaming."""
    path = Path(path)
    with path.open('rb') as stream:
        identity = stream_identity(stream)
    question_only = {key: {'question': row['question']} for key, row in questions.items()}
    expected = {'questions_hash': digest(question_only), 'models': model_lock(c, root),
                'retrieval': c['retrieval'], 'code': source_hash(),
                'mode': c['training']['retrieval_mode'],
                'mode_config': c['training'].get('lexical_pool_k') if c['training']['retrieval_mode'] == 'lexical' else None}
    if os.environ.get('LEGALQA_INDEX_HASH'):
        expected['index_hash'] = os.environ['LEGALQA_INDEX_HASH']
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f'Training retrieval mismatch: {key}')

    def records():
        seen = set()
        with path.open('rb') as stream:
            for key, row in stream_records(stream):
                if key not in questions or key in seen:
                    raise ValueError(f'Unexpected/duplicate training retrieval ID: {key}')
                if row.get('question') != questions[key]['question']:
                    raise ValueError(f'Training retrieval question differs: {key}')
                seen.add(key)
                yield key, row
        if seen != set(questions):
            raise ValueError('Missing query IDs in training retrieval cache')

    return records(), identity
