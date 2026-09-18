"""Import completed lexical retrieval across an audited training-only change."""
import hashlib
import io
import json
import os
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

from .io import ROOT, _unique_pairs, digest, source_hash, write_json
from .training_cache import stream_identity, stream_records


CACHE_NAME = 'train.sft.lexical.retrieval.json'
# Audited against git a574bdb960afb23bff55edb773ab999e40fdc77a and the supplied
# Stage 1 diagnostics. Accept this old code only while retrieval dependencies
# remain unchanged. Do not broadly disable read_retrieval's code check.
LEGACY_CODE = '626e4f377fcfd84dfcb3b51bf2d6af216f2905bbf71e35fcf266eb07cb19a476'
# Pre-streaming Stage 1; retrieval dependencies are identical to the audit below.
PRE_STREAMING_CODES = {
    'b7ce423ad04f159853087bbf1ea6e41e1015d8b66d8d69f5eec7f6d56f247f41',  # Linux: commit 6e1eb19
    'c13dd1e5277994af3a94cd102cdab88fd7f90bd866a421da08ad9b6b1b412882',  # CRLF checkout
}
LEGACY_DEPENDENCIES = {
    # Re-audited after adding opt-in P2 branches. With all new keys absent (the
    # locked v8 config), lexical training retrieval follows the original path.
    'retrieval.py': '72d5652143f5466cd25782e890604d86cc72de267534b295b26da71059ca7c8b',
    'models.py': '7b4e658092a37cfacc9c148fe63390b9c3ab8919786b88e9fa691f343cc9707c',
    'data.py': '0174249f3f06663fff2b5022d1af0764fa0f14452bed29497785ede2d6b70a04',
    'io.py': '0c82e8297a8a611c29bc54bb7fe5e92bbf4f51d6136ca857cffd041733d79435',
    'runtime.py': 'be01f62447f7e55e3f6d2a92612a4004a965bfb4c77ff9f66bc7d587e6709938',
    'phrase_sqlite.py': '1a12c2ade9c806b2cf08f237e6f04e2980c02095362ab2331c3e263a6bd688a2',
}


def compatible_code(code):
    if code == source_hash():
        return True
    return code in {LEGACY_CODE, *PRE_STREAMING_CODES} and all(
        hashlib.sha256((ROOT/'legalqa'/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected
        for name, expected in LEGACY_DEPENDENCIES.items())


@contextmanager
def source_file(source, name):
    source = Path(source)
    if source.is_file() and source.suffix.lower() == '.zip':
        with ZipFile(source) as archive:
            # The diagnostics export uses root-level names. Never extract paths.
            if archive.namelist().count(name) != 1:
                raise ValueError(f'Diagnostics ZIP requires exactly one {name}')
            with archive.open(name) as stream:
                yield stream
    else:
        root = source.parent if source.is_file() else source
        with (root/name).open('rb') as stream:
            yield stream


def source_json(source, name):
    with source_file(source, name) as stream:
        with io.TextIOWrapper(stream, encoding='utf-8-sig') as text:
            return json.load(text, object_pairs_hook=_unique_pairs)


def import_training_retrieval(source, output, questions, c, lock, index_hash):
    output = Path(output)
    if output.exists():
        raise ValueError('Retrieval destination already exists; refusing to overwrite it')
    manifest = source_json(source, 'stage1_manifest.json')
    if manifest.get('schema') != 2 or manifest.get('stage') != 1:
        raise ValueError('Retrieval import requires a Stage 1 schema-2 manifest')
    expected_file = manifest.get('files', {}).get(CACHE_NAME)
    if not expected_file:
        raise ValueError('Snapshot has no completed training retrieval JSON')
    checksum, size = hashlib.sha256(), 0
    with source_file(source, CACHE_NAME) as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            checksum.update(block)
            size += len(block)
    if size != expected_file['size'] or checksum.hexdigest() != expected_file['sha256']:
        raise ValueError('Training retrieval checksum/size differs from the source manifest')
    with source_file(source, CACHE_NAME) as stream:
        identity = stream_identity(stream)
    if identity.get('code') != manifest.get('source_hash') or not compatible_code(identity.get('code')):
        raise ValueError('Retrieval code compatibility has not been verified for this snapshot')
    question_only = {key:{'question':item['question']} for key,item in questions.items()}
    expected = {'questions_hash':digest(question_only), 'models':lock, 'index_hash':index_hash,
                'retrieval':c['retrieval'], 'mode':'lexical',
                'mode_config':c['training'].get('lexical_pool_k')}
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ValueError(f'Training retrieval import mismatch: {key}')
    seen = set()
    with source_file(source, CACHE_NAME) as stream:
        for key, record in stream_records(stream):
            if key not in question_only or key in seen:
                raise ValueError('Training retrieval does not cover exactly the selected training IDs')
            if record.get('question') != question_only[key]['question']:
                raise ValueError(f'Training retrieval question text differs: {key}')
            seen.add(key)
    if seen != set(questions):
        raise ValueError('Training retrieval does not cover exactly the selected training IDs')
    # Preserve all records, contexts, scores and order. Keep original identity
    # as provenance; publish the verified current-code identity atomically.
    provenance = {'source':str(source), 'source_file':CACHE_NAME,
        'source_file_sha256':checksum.hexdigest(), 'source_commit':manifest.get('code_commit'),
        'original_identity':identity, 'compatible_with_code':source_hash(), 'records':len(seen)}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.tmp')
    try:
        with temporary.open('w', encoding='utf-8') as destination:
            destination.write('{"identity":')
            json.dump({**identity, 'code':source_hash()}, destination, ensure_ascii=False, allow_nan=False)
            destination.write(',"import_provenance":')
            json.dump(provenance, destination, ensure_ascii=False, allow_nan=False)
            destination.write(',"records":{')
            with source_file(source, CACHE_NAME) as stream:
                for position, (key, record) in enumerate(stream_records(stream)):
                    if position:
                        destination.write(',')
                    destination.write(json.dumps(key) + ':')
                    json.dump(record, destination, ensure_ascii=False, allow_nan=False)
            destination.write('}}\n')
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Imported lexical retrieval: {len(seen)} records; BM25 skipped; source_sha256={checksum.hexdigest()}',
          flush=True)
    return provenance
