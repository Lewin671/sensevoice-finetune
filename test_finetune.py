"""Tests of the parts of the fine-tuning scripts that decide what is trained on and how it is
scored. They need numpy and zhconv only, no model:

    python3 -m unittest
"""
import json, os, random, tempfile, unittest, zipfile
import label, prepare, sv, train


class AppText(unittest.TestCase):
    def test_normalize_matches_the_app(self):
        # the cases of Local Voice IME's VoiceTextTest.kt
        for raw, expected in [
            (" 今天天气不错。 ", "今天天气不错。"), ("Hello world.", "Hello world."), ("   ", ""),
            ("我在用 Android 手机", "我在用Android手机"), ("你好， 世界", "你好，世界"),
            ("打开 Visual Studio  Code 吧", "打开Visual Studio Code吧"),
            ("你好,world", "你好，world"), ("真的吗?", "真的吗？"), ("OK, 3.5", "OK, 3.5"),
        ]:
            self.assertEqual(expected, sv.app_normalize(raw))


class Scoring(unittest.TestCase):
    def test_tokens_ignore_case_punctuation_and_script_variant(self):
        self.assertEqual(["让", "codex", "看", "3", "遍"], sv.score_tokens("让 Codex 看3遍。"))
        self.assertEqual(sv.score_tokens("後來"), sv.score_tokens("后来"))
        self.assertEqual(["don't", "stop"], sv.score_tokens("Don't stop!"))

    def test_error_counts(self):
        self.assertEqual((0, 4), sv.error_counts("今天，很好。", "今天很好"))
        self.assertEqual((1, 4), sv.error_counts("今天很好", "今天很号"))
        self.assertEqual((1, 3), sv.error_counts("用 mac os", "用 macos os"))
        self.assertEqual((2, 2), sv.error_counts("你好", ""))
        self.assertEqual(3, sv.edit_distance(list("kitten"), list("sitting")))


def utterance(session, n, text, continues=False):
    return {"v": 1, "type": "utterance", "id": f"{session}-{n:02d}", "session": session,
            "audio": f"audio/{session}-{n:02d}.wav", "seconds": 2.0, "text": text, "continues": continues}


class Prepare(unittest.TestCase):
    def test_status_follows_the_last_field_record(self):
        records = [
            utterance("a", 1, "x"),
            {"v": 1, "type": "refined", "id": "a-01", "text": "X"},
            {"v": 1, "type": "field", "session": "a", "dictated": "x。", "text": "y。"},
            {"v": 1, "type": "field", "session": "a", "dictated": "x。", "text": "x。"},
            utterance("b", 1, "x"),
            {"v": 1, "type": "field", "session": "b", "dictated": "x。", "text": "z。"},
            utterance("c", 1, "x"),
            utterance("d", 1, "x"),
            {"v": 1, "type": "field", "session": "d", "dictated": "x。", "text": "z。"},
            {"v": 1, "type": "undone", "session": "d"},
        ]
        by_session, refined = prepare.sessions(records)
        self.assertEqual(["accepted", "corrected", "unconfirmed", "undone"],
                         [prepare.status(s) for s in by_session.values()])
        self.assertEqual({"a-01": "X"}, refined)

    def test_unknown_record_version_is_refused(self):
        with self.assertRaises(ValueError):
            prepare.sessions([{"v": 2, "type": "utterance"}])

    def test_archive_entries_cannot_leave_the_directory(self):
        with tempfile.TemporaryDirectory() as d:
            archive = os.path.join(d, "x.zip")
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../evil.txt", "x")
            with self.assertRaises(ValueError):
                prepare.extract(archive, os.path.join(d, "out"))
            self.assertFalse(os.path.exists(os.path.join(d, "evil.txt")))


def row(uid, **kw):
    base = {"id": uid, "of": 1, "field": None, "status": "unconfirmed", "refined": None}
    return {**base, **kw}


class Labels(unittest.TestCase):
    def test_agreement_needs_every_source(self):
        hyps = {"sensevoice": {"u": "今天去西湖边。"}, "other": {"u": "今天去西湖边"}}
        self.assertEqual("今天去西湖边。", label.agreed(row("u"), hyps))
        self.assertIsNone(label.agreed(row("u", refined="今天去西胡边"), hyps))
        self.assertIsNone(label.agreed(row("u", field="今天去西湖边吧。", status="corrected"), hyps))
        self.assertEqual("今天去西湖边。", label.agreed(row("u", field="今天去西湖边。", status="accepted"), hyps))

    def test_a_single_opinion_is_not_agreement(self):
        self.assertIsNone(label.agreed(row("u"), {"sensevoice": {"u": "你好。"}}))
        self.assertIsNone(label.agreed(row("u", refined=""), {"sensevoice": {"u": ""}}))

    def test_field_of_a_session_with_several_utterances_is_not_compared(self):
        hyps = {"sensevoice": {"u": "你好。"}, "other": {"u": "你好"}}
        self.assertEqual("你好。", label.agreed(row("u", of=2, field="你好，世界。", status="accepted"), hyps))

    def test_undone_field_is_not_evidence(self):
        hyps = {"sensevoice": {"u": "你好。"}, "other": {"u": "你好"}}
        self.assertEqual("你好。", label.agreed(row("u", field="再见。", status="undone"), hyps))

    def test_reviewed_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, encoding="utf-8") as f:
            f.write("# comment\na\tA\t你好。\nb\tX\t\n")
        self.assertEqual({"a": ("A", "你好。"), "b": ("X", "")}, label.read_manual(f.name))
        with open(f.name, "a", encoding="utf-8") as g:
            g.write("a\tB\t再见。\n")
        with self.assertRaises(ValueError):
            label.read_manual(f.name)
        os.unlink(f.name)

    def test_retries_and_sessions_stay_in_one_fold(self):
        rows = [{"session": f"s{i}", "text": t, "seconds": 2.0} for i, t in enumerate(
            ["这张地铁图怎么看？", "这张地铁图怎么看", "明天早上八点出发。", "完全不同的一句话。"])]
        rows.append({"session": "s3", "text": "同一个会话的另一句。", "seconds": 2.0})
        groups = label.groups(rows)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[3], groups[4])
        self.assertEqual(3, len(set(groups)))
        for r, g in zip(rows, groups):
            r["group"] = g
        label.assign_folds(rows, 3, seed=0)
        self.assertEqual(rows[0]["fold"], rows[1]["fold"])
        self.assertEqual(rows[3]["fold"], rows[4]["fold"])
        self.assertEqual({0, 1, 2}, {r["fold"] for r in rows})


class Training(unittest.TestCase):
    def test_join_of_a_sentence_that_went_on(self):
        self.assertEqual("因为昨天下雨所以没有出门。",
                         train.join_texts(["因为昨天下雨所以。", "没有出门。"], True))
        self.assertEqual("He opened the window looking at the rain.",
                         train.join_texts(["He opened the window.", "Looking at the rain."], True))
        self.assertEqual("The kettle is boiling I will make some tea.",
                         train.join_texts(["The kettle is boiling.", "I will make some tea."], True))

    def test_join_of_separate_sentences(self):
        self.assertEqual("你好。再见。", train.join_texts(["你好。", "再见。"], False))
        self.assertEqual("It is late. Go home.", train.join_texts(["It is late.", "Go home."], False))
        self.assertEqual("你好。Go home.", train.join_texts(["你好。", "Go home."], False))

    def test_items_leave_the_held_out_fold_alone(self):
        labels = [
            {"id": "a-01", "audio": "a1", "text": "一。", "seconds": 1, "fold": 0, "grade": "A", "joins": []},
            {"id": "a-02", "audio": "a2", "text": "二。", "seconds": 1, "fold": 0, "grade": "B", "joins": ["a-01"]},
            {"id": "b-01", "audio": "b1", "text": "三。", "seconds": 1, "fold": 1, "grade": "A", "joins": []},
            {"id": "b-02", "audio": "b2", "text": "四。", "seconds": 1, "fold": 1, "grade": "A", "joins": ["b-01"]},
        ]
        tr, held = train.build_items(labels, 0)
        self.assertEqual([["a-01"]], [i["ids"] for i in held])  # grade B is not tested on
        self.assertEqual([["b-01"], ["b-02"], ["b-01", "b-02"]], [i["ids"] for i in tr])
        self.assertEqual("三四。", tr[2]["text"])
        tr, held = train.build_items(labels, -1)
        self.assertEqual(6, len(tr))
        self.assertEqual([], held)

    def test_batches_cover_everything_within_the_limit(self):
        rng = random.Random(0)
        lengths = [rng.randint(5, 300) for _ in range(200)]
        seen = []
        for b in train.batches(lengths, 480, 24, rng):
            self.assertLessEqual(len(b), 24)
            if len(b) > 1:
                self.assertLessEqual(max(lengths[i] for i in b) * len(b), 480 + 300)
            seen += b
        self.assertEqual(sorted(seen), list(range(200)))


if __name__ == "__main__":
    unittest.main()
