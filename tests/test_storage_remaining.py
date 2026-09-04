from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from legalqa_baseline.storage import SearchIndex, build_index, iter_contexts


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _passage(marker: str) -> str:
    return f"{marker} " + "nội dung pháp lý hợp lệ " * 5


class StorageRemainingRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.train_path = self.root / "train.json"
        _write_json(
            self.train_path,
            {"1": {"question": "Câu hỏi kiểm thử?", "answer": "Câu trả lời."}},
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _build(self, contexts: Path, database: Path, **kwargs: object) -> None:
        build_index(contexts, self.train_path, database, **kwargs)  # type: ignore[arg-type]

    def test_iter_contexts_directory_and_zip_use_the_same_filter_and_order(self) -> None:
        # Bản thân source có thể nằm trong thư mục ẩn; chỉ các thành phần tương
        # đối bên trong source mới được dùng để lọc.
        contexts = self.root / ".contexts"
        _write_json(
            contexts / "nested" / "context_2.json",
            {"id": "two", "name": "Two", "link": "", "passage": _passage("two")},
        )
        _write_json(
            contexts / "context_1.json",
            {"id": "one", "name": "One", "link": "", "passage": _passage("one")},
        )
        _write_json(
            contexts / "metadata.json",
            {
                "id": "manifest",
                "name": "Not a context",
                "link": "",
                "passage": _passage("manifest"),
            },
        )
        _write_json(
            contexts / "__MACOSX" / "context_ignored.json",
            {
                "id": "ignored",
                "name": "Archive metadata",
                "link": "",
                "passage": _passage("ignored"),
            },
        )

        archive_path = self.root / "contexts.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for item_path in contexts.rglob("*.json"):
                archive.write(item_path, item_path.relative_to(contexts).as_posix())

        directory_ids = [str(item["id"]) for item in iter_contexts(contexts)]
        archive_ids = [str(item["id"]) for item in iter_contexts(archive_path)]
        self.assertEqual(directory_ids, ["one", "two"])
        self.assertEqual(archive_ids, directory_ids)

    def test_context_schema_ids_and_duplicates_are_validated(self) -> None:
        invalid_cases = {
            "non_object": ["not", "an", "object"],
            "missing_id": {
                "name": "Missing",
                "link": "",
                "passage": _passage("missing"),
            },
            "blank_id": {
                "id": "   ",
                "name": "Blank",
                "link": "",
                "passage": _passage("blank"),
            },
            "invalid_passage": {
                "id": "bad-passage",
                "name": "Bad passage",
                "link": "",
                "passage": None,
            },
            "invalid_name": {
                "id": "bad-name",
                "name": 123,
                "link": "",
                "passage": _passage("bad-name"),
            },
            "invalid_link": {
                "id": "bad-link",
                "name": "Bad link",
                "link": [],
                "passage": _passage("bad-link"),
            },
        }
        for case_name, payload in invalid_cases.items():
            with self.subTest(case=case_name):
                contexts = self.root / f"contexts_{case_name}"
                _write_json(contexts / "context_1.json", payload)
                database = self.root / f"{case_name}.sqlite"
                with self.assertRaises(ValueError):
                    self._build(contexts, database)
                self.assertFalse(database.exists())

        duplicate_contexts = self.root / "contexts_duplicate"
        _write_json(
            duplicate_contexts / "context_1.json",
            {"id": 7, "name": "First", "link": "", "passage": _passage("first")},
        )
        _write_json(
            duplicate_contexts / "context_2.json",
            {"id": "7", "name": "Second", "link": "", "passage": _passage("second")},
        )
        duplicate_database = self.root / "duplicate.sqlite"
        with self.assertRaises(ValueError):
            self._build(duplicate_contexts, duplicate_database)
        self.assertFalse(duplicate_database.exists())

        numeric_contexts = self.root / "contexts_numeric"
        _write_json(
            numeric_contexts / "context_1.json",
            {"id": 123, "name": "Numeric", "link": "", "passage": _passage("numericmarker")},
        )
        numeric_database = self.root / "numeric.sqlite"
        self._build(numeric_contexts, numeric_database)
        with SearchIndex(numeric_database) as index:
            matches = index.search_contexts("numericmarker", top_k=1)
        self.assertEqual(matches[0]["context_id"], "123")

    def test_invalid_chunk_configuration_and_zero_chunks_never_publish(self) -> None:
        empty_contexts = self.root / "empty_contexts"
        empty_contexts.mkdir()

        invalid_database = self.root / "invalid.sqlite"
        with self.assertRaises(ValueError):
            self._build(
                empty_contexts,
                invalid_database,
                max_chunk_words=0,
                overlap_words=0,
            )
        self.assertFalse(invalid_database.exists())

        zero_database = self.root / "zero.sqlite"
        with self.assertRaises(ValueError):
            self._build(empty_contexts, zero_database)
        self.assertFalse(zero_database.exists())

        contexts = self.root / "force_contexts"
        context_path = contexts / "context_1.json"
        _write_json(
            context_path,
            {"id": "old", "name": "Old", "link": "", "passage": _passage("oldmarker")},
        )
        database = self.root / "force.sqlite"
        self._build(contexts, database)

        _write_json(
            context_path,
            {"id": "new", "name": "New", "link": "", "passage": "   "},
        )
        with self.assertRaises(ValueError):
            self._build(contexts, database, force=True)
        with self.assertRaises(ValueError):
            self._build(
                contexts,
                database,
                max_chunk_words=0,
                overlap_words=0,
                force=True,
            )

        with SearchIndex(database) as index:
            old_matches = index.search_contexts("oldmarker", top_k=1)
            new_matches = index.search_contexts("newmarker", top_k=1)
        self.assertEqual(old_matches[0]["context_id"], "old")
        self.assertEqual(new_matches, [])

    def test_corpus_hash_has_unambiguous_field_boundaries(self) -> None:
        hashes: list[str] = []
        variants = (
            {
                "id": "same",
                "name": "N",
                "link": "B\x00C",
                "passage": "D " + "word " * 18,
            },
            {
                "id": "same",
                "name": "N",
                "link": "B",
                "passage": "C\x00D " + "word " * 18,
            },
        )
        for index_number, context in enumerate(variants):
            contexts = self.root / f"hash_contexts_{index_number}"
            _write_json(contexts / "context_1.json", context)
            database = self.root / f"hash_{index_number}.sqlite"
            self._build(contexts, database)
            with SearchIndex(database) as index:
                hashes.append(index.metadata()["corpus_sha256"])

        self.assertNotEqual(hashes[0], hashes[1])

    def test_search_contexts_rejects_non_positive_top_k_before_querying(self) -> None:
        contexts = self.root / "search_contexts"
        _write_json(
            contexts / "context_1.json",
            {"id": "one", "name": "One", "link": "", "passage": _passage("searchmarker")},
        )
        database = self.root / "search.sqlite"
        self._build(contexts, database)

        with SearchIndex(database) as index:
            for question in ("searchmarker", "   "):
                for top_k in (0, -1):
                    with self.subTest(question=question, top_k=top_k):
                        with self.assertRaises(ValueError):
                            index.search_contexts(question, top_k=top_k)

    def test_search_contexts_prioritizes_controlled_exact_phrase_alias(self) -> None:
        contexts = self.root / "controlled_aliases"
        _write_json(
            contexts / "context_exact.json",
            {
                "id": "power-plan-viii",
                "name": "Quy hoạch điện VIII",
                "link": "",
                "passage": (
                    "Thủ tướng Chính phủ chính thức phê duyệt Quy hoạch điện VIII. "
                    "Nội dung quy hoạch nguồn điện và lưới điện quốc gia."
                ),
            },
        )
        _write_json(
            contexts / "context_noise.json",
            {
                "id": "generic-approval",
                "name": "Thông tin phê duyệt",
                "link": "",
                "passage": (
                    "Thủ tướng chính thức phê duyệt nhiều dự án điện. "
                    "Thông tin chung về quy hoạch và quyết định đầu tư."
                ),
            },
        )
        database = self.root / "controlled_aliases.sqlite"
        self._build(contexts, database)

        with SearchIndex(database) as index:
            matches = index.search_contexts(
                "Thủ tướng chính thức phê duyệt Quy hoạch điện 8?",
                top_k=2,
            )

        self.assertEqual(matches[0]["context_id"], "power-plan-viii")
        self.assertGreater(int(matches[0]["exact_phrase_matches"]), 0)

    def test_smoke_query_priorities_rescue_exact_governing_sections(self) -> None:
        contexts = self.root / "smoke_query_priorities"
        fixtures = (
            (
                "pccc-report",
                "Thông tư 17/2021/TT-BCA",
                "Điều 10. Thống kê, báo cáo công tác quản lý, bảo quản, bảo dưỡng "
                "phương tiện phòng cháy, chữa cháy. Trình tự báo cáo và cơ quan "
                "tiếp nhận báo cáo được xác định theo đơn vị trực tiếp quản lý "
                "phương tiện và phạm vi quản lý của cơ quan công an có thẩm quyền.",
            ),
            (
                "pccc-noise",
                "Quy định phòng cháy chữa cháy",
                "Cơ sở lập phương án phòng cháy chữa cháy, tổ chức thực tập, báo cáo "
                "kết quả và quản lý nhiều loại phương tiện theo kế hoạch hằng năm. "
                "Nội dung chung này không quy định nơi nhận báo cáo bảo dưỡng.",
            ),
            (
                "fund-article-42",
                "Luật An toàn, vệ sinh lao động",
                "Điều 42. Sử dụng Quỹ bảo hiểm tai nạn lao động, bệnh nghề nghiệp. "
                "Quỹ chi trả phí giám định, trợ cấp, hỗ trợ phòng ngừa và phục hồi "
                "chức năng lao động cho người lao động theo các khoản của điều này.",
            ),
            (
                "fund-noise",
                "Chi phí quản lý bảo hiểm",
                "Cơ quan báo cáo việc quản lý và sử dụng quỹ bảo hiểm xã hội, quỹ "
                "bảo hiểm tai nạn lao động và kinh phí hành chính theo dự toán. "
                "Nội dung tập trung vào quyết toán chi phí quản lý hằng năm.",
            ),
        )
        for context_id, name, passage in fixtures:
            _write_json(
                contexts / f"context_{context_id}.json",
                {"id": context_id, "name": name, "link": "", "passage": passage},
            )
        database = self.root / "smoke_query_priorities.sqlite"
        self._build(contexts, database)

        with SearchIndex(database) as index:
            pccc_matches = index.search_contexts(
                "Thực hiện báo cáo phương tiện phòng cháy chữa cháy như thế nào "
                "và tại đâu?",
                top_k=4,
            )
            fund_matches = index.search_contexts(
                "Sử dụng Quỹ bảo hiểm tai nạn lao động, bệnh nghề nghiệp?",
                top_k=4,
            )

        self.assertEqual(pccc_matches[0]["context_id"], "pccc-report")
        self.assertGreater(int(pccc_matches[0]["exact_phrase_matches"]), 0)
        self.assertEqual(fund_matches[0]["context_id"], "fund-article-42")
        self.assertGreater(int(fund_matches[0]["exact_phrase_matches"]), 0)

    def test_hash_and_bm25_ties_are_independent_of_filename_order(self) -> None:
        databases: list[Path] = []
        for variant, assignments in enumerate(
            (
                (("context_1.json", "alpha"), ("context_2.json", "beta")),
                (("context_1.json", "beta"), ("context_2.json", "alpha")),
            )
        ):
            contexts = self.root / f"ordered_contexts_{variant}"
            for filename, context_id in assignments:
                _write_json(
                    contexts / filename,
                    {
                        "id": context_id,
                        "name": "Same name",
                        "link": "",
                        "passage": _passage("sharedmarker"),
                    },
                )
            database = self.root / f"ordered_{variant}.sqlite"
            self._build(contexts, database)
            databases.append(database)

        hashes: list[str] = []
        result_orders: list[list[str]] = []
        for database in databases:
            with SearchIndex(database) as index:
                hashes.append(index.metadata()["corpus_sha256"])
                result_orders.append(
                    [
                        str(item["context_id"])
                        for item in index.search_contexts("sharedmarker", top_k=10)
                    ]
                )

        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(result_orders[0], ["alpha", "beta"])
        self.assertEqual(result_orders[1], result_orders[0])


if __name__ == "__main__":
    unittest.main()
