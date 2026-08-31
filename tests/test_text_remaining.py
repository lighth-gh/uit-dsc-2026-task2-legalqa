from __future__ import annotations

import unittest

from legalqa_baseline.text import chunk_passage


class ChunkPassageRemainingRegressionTests(unittest.TestCase):
    def test_short_final_window_preserves_every_source_token(self) -> None:
        source_tokens = [f"word{index}" for index in range(101)]

        chunks = chunk_passage(
            " ".join(source_tokens),
            max_words=100,
            overlap_words=0,
        )

        covered_tokens = {
            token
            for chunk in chunks
            for token in chunk.split()
        }
        self.assertEqual(covered_tokens, set(source_tokens))
        self.assertTrue(all(len(chunk.split()) <= 100 for chunk in chunks))
        self.assertEqual(len(chunks), 2)
        self.assertEqual(set(chunks[0].split()).intersection(chunks[1].split()), set())
        self.assertEqual(
            chunks[0].split() + chunks[1].split(),
            source_tokens,
        )

    def test_invalid_min_words_is_rejected(self) -> None:
        for min_words in (0, -1, 11):
            with self.subTest(min_words=min_words):
                with self.assertRaises(ValueError):
                    chunk_passage(
                        "one two three",
                        max_words=10,
                        overlap_words=0,
                        min_words=min_words,
                    )

    def test_twenty_word_preamble_is_preserved(self) -> None:
        preamble_tokens = ["pham_vi_ap_dung"] + [
            f"preamble{index}" for index in range(19)
        ]
        passage = (
            " ".join(preamble_tokens)
            + "\nĐiều 1. "
            + "Nội dung quy định của điều này. " * 6
        )

        chunks = chunk_passage(passage)
        covered_tokens = {
            token
            for chunk in chunks
            for token in chunk.split()
        }

        self.assertTrue(set(preamble_tokens).issubset(covered_tokens))

    def test_generic_passage_shorter_than_min_words_is_preserved(self) -> None:
        passage = "Quy định ngắn nhưng vẫn có ý nghĩa pháp lý."

        chunks = chunk_passage(passage)

        self.assertIn(passage, "\n".join(chunks))

    def test_article_heading_with_narrow_no_break_space_starts_a_chunk(self) -> None:
        preamble = " ".join(f"preamble{index}" for index in range(30))
        short_article = "Điều\u202f1. Cấm hút thuốc."

        chunks = chunk_passage(f"{preamble}\n{short_article}")
        normalized_chunks = [chunk.replace("\u202f", " ") for chunk in chunks]

        self.assertTrue(
            any(chunk.startswith("Điều 1.") for chunk in normalized_chunks),
            normalized_chunks,
        )


if __name__ == "__main__":
    unittest.main()
