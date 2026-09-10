import ast
import json
import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from legalqa.data import iter_documents, legal_parents, document_number, prepare, split_records, token_children
from legalqa.generation import package_submission
from legalqa.io import Journal, ROOT, config, group_key, read_json, validate_predictions, write_json
from legalqa.models import require_approved
from legalqa.metrics import select_reports
from legalqa.prompts import answer_flags, clean_answer, complete_truncated_answer, pack_prompt, window_around_seed
from legalqa.retrieval import Retriever, diversified, retrieval_adjustment, rrf
from legalqa.training import training_examples


class TinyTokenizer:
    """Deterministic tokenizer only for CPU control-flow tests, never for model training."""
    eos_token_id = 1
    pad_token_id = 0
    def __init__(self):
        self.words = {}
        self.inverse = {}
    def __call__(self,text,**kwargs):
        matches = list(re.finditer(r"\S+",text))
        ids = []
        for m in matches:
            if m.group() not in self.words:
                token = len(self.words)+2
                self.words[m.group()] = token
                self.inverse[token] = m.group()
            ids.append(self.words[m.group()])
        return {"input_ids":ids,"offset_mapping":[m.span() for m in matches]}
    def decode(self,ids,**kwargs):
        return " ".join(self.inverse.get(i,"") for i in ids)
    def apply_chat_template(self,messages,**kwargs):
        return self(" ".join(m["content"] for m in messages)+" ASSISTANT")["input_ids"]


class CoreTests(unittest.TestCase):
    def test_select_uses_meteor_and_rejects_incomparable_references(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            a={"label":"a","meteor":.4,"rougeL":.6,"reference_hash":"same",
               "metric_identity":{"implementation":"fixture"},"per_question":{"id":{}}}
            b={**a,"label":"b","meteor":.41,"rougeL":.5}
            write_json(d/"a.json",a);write_json(d/"b.json",b)
            best=select_reports([d/"a.json",d/"b.json"],d/"selected.json")
            self.assertEqual(best["label"],"b")
            b["reference_hash"]="different"
            write_json(d/"b.json",b)
            with self.assertRaises(ValueError):select_reports([d/"a.json",d/"b.json"],d/"bad.json")

    def test_checkpoint_recovers_interrupted_last_line_and_rejects_stale_config(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"checkpoint.jsonl"
            journal=Journal(path,{"seed":2026})
            journal.append("a",{"answer":"đáp án"})
            with path.open("ab") as f:f.write(b'{"id":"unfinished')
            resumed=Journal(path,{"seed":2026})
            self.assertEqual(list(resumed.records),["a"])
            resumed.append("b",{"answer":"tiếp tục"})
            self.assertEqual(len(Journal(path,{"seed":2026}).records),2)
            with self.assertRaises(ValueError):Journal(path,{"seed":42})

    def test_approval_and_budget_config(self):
        c = config()
        require_approved(c)
        self.assertEqual(c["models"]["generator"],"AITeamVN/Vi-Qwen2-3B-RAG")
        c["models"]["generator"] = "Qwen/Qwen3-4B"
        with self.assertRaises(ValueError):require_approved(c)

    def test_duplicate_questions_and_answers_do_not_cross_splits(self):
        qa = {str(i):{"question":f"câu hỏi {i}","answer":f"đáp án số {i}"} for i in range(50)}
        qa["1"]["question"] = "  CÂU HỎI 0? "
        qa["2"]["answer"] = qa["0"]["answer"]
        parts,_ = split_records(qa)
        destinations = {k:p for p,ids in parts.items() for k in ids}
        self.assertEqual(destinations["0"],destinations["1"])
        self.assertEqual(destinations["0"],destinations["2"])
        self.assertEqual(split_records(qa),split_records(dict(reversed(list(qa.items())))))
        self.assertEqual(set(destinations),set(qa))

    def test_different_numbers_remain_different(self):
        self.assertNotEqual(group_key("phạt 10 triệu?"),group_key("phạt 100 triệu?"))

    def test_duplicate_json_keys_raise(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"bad.json";p.write_text('{"1":1,"1":2}')
            with self.assertRaises(ValueError):read_json(p)

    def test_nested_corpus_zip_is_read_without_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"context.zip"
            with ZipFile(p,"w") as z:
                z.writestr("selected-contexts/selected-contexts/context_1.json",json.dumps({"id":1,"passage":"Điều 1. Quy định\nNội dung","name":None}))
            docs = list(iter_documents(p))
            self.assertEqual(docs[0]["doc_id"],"1")
            self.assertEqual(docs[0]["name"],"")

    def test_article_parents_preserve_amendment_and_no_page_id_citation(self):
        d = {"doc_id":"396608","text":"Số: 5868/QĐ-BYT\nĐiều 1. Nhiệm vụ\nA\nĐiều 2. Hiệu lực\nB",
             "source_file":"context_1.json","link":"https://test/396608.aspx"}
        parents = list(legal_parents(d))
        self.assertEqual(len(parents),3)
        self.assertEqual(parents[1]["number"],"5868/QĐ-BYT")
        self.assertEqual(document_number("https://test/396608.aspx"),"")
        self.assertIn("Điều 2.",parents[2]["text"])

    def test_child_windows_have_no_gaps_and_stop_at_parent_boundary(self):
        tok = TinyTokenizer()
        text = " ".join(f"w{i}" for i in range(70))
        p = {"parent_id":"1:0","text":text,"heading":"Điều 1","number":""}
        chunks = list(token_children(p,tok,20,5,8))
        covered = set()
        for child in chunks:
            self.assertLessEqual(len(tok(child["text"])["input_ids"]),20)
            covered.update(child["text"].split())
        self.assertEqual(covered,set(text.split()))
        self.assertEqual(chunks[-1]["end"],len(text))

    def test_fts5_handles_vietnamese_and_ascending_bm25(self):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE s USING fts5(text, tokenize='unicode61 remove_diacritics 0')")
        con.executemany("INSERT INTO s(text) VALUES(?)",[("đấu thầu đấu thầu bảo lãnh",),("câu không liên quan",),("bảo lãnh",)])
        rows = con.execute('SELECT text FROM s WHERE s MATCH ? ORDER BY bm25(s)',('"đấu" OR "thầu"',)).fetchall()
        self.assertEqual(len(rows),1)
        self.assertIn("đấu thầu",rows[0][0])
        con.close()

    def test_phrase_and_precise_bm25_recover_specific_legal_context(self):
        con = sqlite3.connect(":memory:")
        con.execute("CREATE VIRTUAL TABLE search USING fts5(header, text, tokenize='unicode61 remove_diacritics 0')")
        con.executemany("INSERT INTO search(header,text) VALUES(?,?)", [
            ("94/2013/NĐ-CP", "danh mục phao tròn cứu sinh dự trữ quốc gia"),
            ("QCVN 05:2016/BTC", "đơn vị trực tiếp quản lý phao tròn cứu sinh chuẩn bị đầy đủ vật tư thiết bị dụng cụ"),
            ("", "nội dung không liên quan"),
        ])
        engine = Retriever.__new__(Retriever)
        engine.con = con
        engine.c = {"retrieval":{"precise_bm25_k":10}}
        question = "Đơn vị quản lý phao tròn cứu sinh chuẩn bị vật tư thiết bị dụng cụ như thế nào?"
        self.assertEqual(engine.bm25_phrases(question)[0], 2)
        self.assertEqual(engine.bm25_precise(question)[0], 2)
        con.close()

    def test_rrf_and_parent_diversity(self):
        self.assertEqual(rrf([[1,2],[2,3]])[0],2)
        chunks = {1:{"parent_id":"a"},2:{"parent_id":"a"},3:{"parent_id":"b"}}
        self.assertEqual(diversified([1,2,3],chunks,2,1),[1,3])

    def test_context_window_contains_late_retrieval_hit(self):
        tok = TinyTokenizer();text = " ".join(f"word{i}" for i in range(200))
        start = text.index("word180")
        parent = {"text":text,"seed_start":start,"seed_end":start+len("word180")}
        result = window_around_seed(parent,tok,30)
        self.assertIn("word180",result)
        self.assertLessEqual(len(result.split()),30)

    def test_prompt_budget_and_train_answer_mask(self):
        c = config();tok = TinyTokenizer()
        c["generation"]["max_input_tokens"] = 700
        qa = {"1":{"question":"câu hỏi","answer":"Đáp án gốc 123/2020/NĐ-CP, không thay đổi."}}
        parents = [{"parent_id":"a","text":" ".join(f"luật{i}" for i in range(500)),"seed_start":0,"seed_end":20}]
        ids,packed = pack_prompt(qa["1"]["question"],parents,tok,c)
        self.assertLessEqual(len(ids),700)
        samples,report = training_examples(qa,{"1":{"contexts":parents}},tok,c)
        labels = samples[0]["labels"]
        answer_ids = tok(qa["1"]["answer"])["input_ids"]+[tok.eos_token_id]
        self.assertEqual(labels[-len(answer_ids):],answer_ids)
        self.assertTrue(all(x==-100 for x in labels[:-len(answer_ids)]))
        self.assertEqual(report["used"],1)

    def test_prompt_gives_top_context_more_room(self):
        c = config();tok = TinyTokenizer()
        c["generation"].update({"max_input_tokens":900,"parent_max_tokens":400,"min_context_tokens":80})
        parents = [{"parent_id":str(i),"text":" ".join(f"p{i}w{j}" for j in range(500)),
                    "seed_start":0,"seed_end":20} for i in range(4)]
        _,packed = pack_prompt("câu hỏi pháp luật",parents,tok,c)
        sizes = [len(tok(item["text"])["input_ids"]) for item in packed]
        self.assertGreater(sizes[0],sizes[1])
        self.assertGreater(sizes[1],sizes[2])
        self.assertGreaterEqual(sizes[-1],32)

    def test_cleaning_preserves_law_numbers_and_money(self):
        text = "**Trả lời:**\n- Theo Điều 2 Nghị định 12/2020/NĐ-CP, phạt 1.000.000 đồng."
        clean = clean_answer(text)
        self.assertIn("12/2020/NĐ-CP",clean)
        self.assertIn("1.000.000",clean)
        self.assertNotIn("**",clean)
        self.assertFalse(answer_flags(clean,clean)["unsupported_document_numbers"])
        self.assertTrue(answer_flags("Theo 99/2025/NĐ-CP.",clean)["unsupported_document_numbers"])

    def test_answer_flags_ignore_dates_and_substantive_information_clauses(self):
        evidence = "Điều 1. Áp dụng từ ngày 01/08/2022 theo 12/2020/NĐ-CP."
        answer = "Từ 01/08/2022, nếu việc xác minh không đủ thông tin thì cơ quan hải quan được từ chối ưu đãi."
        self.assertFalse(answer_flags(answer,evidence)["unsupported_document_numbers"])
        self.assertFalse(answer_flags(answer,evidence)["refusal"])
        self.assertTrue(answer_flags("Không có thông tin để trả lời.",evidence)["refusal"])

    def test_complete_truncated_answer_drops_incomplete_tail(self):
        completed = " ".join(["nội dung"]*24)+"."
        answer = completed+" Phần tiếp theo đang bị cắt giữa"
        self.assertEqual(complete_truncated_answer(answer),completed)

    def test_retrieval_adjustment_rewards_exact_document_phrase_and_year(self):
        settings = {"lexical_score_weight":2.0,"phrase_match_bonus":1.0,
                    "exact_document_bonus":4.0,"year_match_bonus":1.0,"recency_bonus":.75}
        question = "Quy định mới nhất năm 2023 tại 06/2023/TT-BVHTTDL về chuyên viên di sản văn hóa?"
        exact = {"header":"06/2023/TT-BVHTTDL", "text":"quy định về chuyên viên di sản văn hóa"}
        old = {"header":"16/2021/TT-BVHTTDL", "text":"quy định về viên chức di sản"}
        self.assertGreater(retrieval_adjustment(question,exact,settings,2023),
                           retrieval_adjustment(question,old,settings,2023)+4)

    def test_three_kaggle_notebooks_share_quality_config_and_artifacts(self):
        artifact = "/kaggle/input/datasets/lighth/ver3-smoke-output/legalqa_smoke_full_v1"
        dataset = "/kaggle/input/datasets/lighth/uit-dsc-2026-task2-legalqa-train"
        for name in ["legalqa_smoke_pipeline.ipynb", "legalqa_dev100_pipeline.ipynb",
                     "legalqa_main_run.ipynb"]:
            notebook = json.loads((ROOT/name).read_text(encoding="utf-8"))
            source = "\n".join("".join(cell.get("source",[])) for cell in notebook["cells"])
            self.assertIn(artifact,source,name)
            self.assertIn(dataset,source,name)
            self.assertIn("CODE / 'config.json'",source,name)
            self.assertNotIn("['retrieval'].update",source,name)
            self.assertNotIn("['generation'].update",source,name)

    def test_submission_rejects_wrong_id_set_even_same_length(self):
        with self.assertRaises(ValueError):validate_predictions({"x":{"answer":"a"}},{"y":{}})

    def test_submission_rejects_nonstring_empty_and_extra_fields(self):
        for item in [{"answer":None},{"answer":" "},{"answer":"a","question":"q"}]:
            with self.assertRaises(ValueError):validate_predictions({"1":item},{"1":{}})

    def test_prepare_and_package_end_to_end_without_models(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            qa = {str(i):{"question":f"câu hỏi {i}","answer":f"đáp án {i}"} for i in range(30)}
            write_json(d/"train.json",qa);write_json(d/"public.json",{"101":{"question":"câu hỏi công khai"}})
            prepare(d/"train.json",d/"public.json",d/"data",2026)
            prepare(d/"train.json",d/"public.json",d/"data",2026)
            public = read_json(d/"data/test.questions.json")
            self.assertNotIn("answer",public["101"])
            write_json(d/"pred.json",{"101":{"answer":"một câu trả lời thử cho kiểm định schema"}})
            package_submission(d/"pred.json",d/"data/test.questions.json",d/"out.zip")
            with ZipFile(d/"out.zip") as z:
                self.assertEqual(z.namelist(),["submission.json"])
            with self.assertRaises(ValueError):package_submission(d/"pred.json",d/"data/test.questions.json",d/"bad.zip","../x.json")

    def test_official_scorer_is_present_and_uses_asymmetric_schema(self):
        text = (ROOT/"vendor/scoring.py").read_text()
        tree = ast.parse(text)
        function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="eval_qa")
        self.assertIn("v['answer']",ast.get_source_segment(text,function))
        self.assertIn("use_stemmer=False",text)
        self.assertIn(".split()",text)
        self.assertIn('[^a-z0-9]+',(ROOT/"vendor/rouge_score/tokenize.py").read_text())


if __name__=="__main__":
    unittest.main()
