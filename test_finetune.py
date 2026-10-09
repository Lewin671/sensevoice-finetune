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

    def test_archive_is_not_written_through_a_link(self):
        # zipfile drops ".." where realpath follows the link first: check what is written
        with tempfile.TemporaryDirectory() as d:
            out, elsewhere = os.path.join(d, "out"), os.path.join(d, "elsewhere", "deep")
            os.makedirs(out), os.makedirs(elsewhere)
            os.symlink(elsewhere, os.path.join(out, "link"))
            for name in ("link/../../out/payload", "link/payload"):
                archive = os.path.join(d, "x.zip")
                with zipfile.ZipFile(archive, "w") as z:
                    z.writestr(name, "x")
                with self.assertRaises(ValueError):
                    prepare.extract(archive, out)
            self.assertEqual([], [f for _, _, files in os.walk(os.path.join(d, "elsewhere")) for f in files])

    def test_archive_that_is_too_large_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            archive = os.path.join(d, "x.zip")
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("log.jsonl", "0" * 4096)
            limit, prepare.MAX_BYTES = prepare.MAX_BYTES, 1024
            try:
                with self.assertRaises(ValueError):
                    prepare.extract(archive, os.path.join(d, "out"))
            finally:
                prepare.MAX_BYTES = limit
            with open(os.path.join(prepare.extract(archive, os.path.join(d, "ok")), "log.jsonl")) as f:
                self.assertEqual("0" * 4096, f.read())


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
            ["这班地铁开往哪里？", "这班地铁开往哪里", "明天早上八点出发。", "完全不同的一句话。"])]
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


    def test_a_sentence_said_in_parts_and_in_one_go_stays_in_one_fold(self):
        rows = [
            {"id": "a-01", "session": "a", "text": "今天天气特别晴朗。", "seconds": 2.0, "joins": []},
            {"id": "a-02", "session": "a", "text": "我们一起去公园散步。", "seconds": 2.0, "joins": ["a-01"]},
            {"id": "b-01", "session": "b", "text": "今天天气特别晴朗我们一起去公园散步。", "seconds": 4.0, "joins": []},
            {"id": "c-01", "session": "c", "text": "完全不同的一句话。", "seconds": 2.0, "joins": []},
        ]
        groups = label.groups(rows)
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[0], groups[3])


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

    def test_a_sentence_is_not_joined_when_a_part_was_filtered_out(self):
        labels = [
            {"id": "a-02", "audio": "a2", "text": "二。", "seconds": 1, "fold": 1, "grade": "A", "joins": ["a-01"]},
        ]
        tr, _ = train.build_items(labels, 0)
        self.assertEqual([["a-02"]], [i["ids"] for i in tr])

    def test_schedule_follows_progress(self):
        self.assertLess(train.schedule(0.0), 0.05)
        self.assertAlmostEqual(1.0, train.schedule(0.1))
        self.assertGreater(train.schedule(0.5), 0.5)  # still learning half way through
        self.assertAlmostEqual(0.1, train.schedule(1.0))
        values = [train.schedule(p / 100) for p in range(10, 101)]
        self.assertEqual(sorted(values, reverse=True), values)

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


class Scores(unittest.TestCase):
    def run_score(self, d, *args):
        import subprocess, sys
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.run([sys.executable, os.path.join(here, "score.py"), *args], cwd=d,
                              capture_output=True, text=True)

    def test_a_missing_transcript_is_not_dropped_silently(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "m.jsonl"), "w", encoding="utf-8") as f:
                for i, who in enumerate("xxy"):
                    f.write(json.dumps({"id": f"u{i}", "text": "你好", "group": who}, ensure_ascii=False) + "\n")
            for name, hyp in (("a", {"u0": "你好", "u1": "你号", "u2": "你好"}), ("b", {"u0": "你好", "u1": "你好"})):
                with open(os.path.join(d, name + ".json"), "w", encoding="utf-8") as f:
                    json.dump(hyp, f, ensure_ascii=False)
            refused = self.run_score(d, "--manifest", "m.jsonl", "a=a.json", "b=b.json")
            self.assertNotEqual(0, refused.returncode)
            self.assertIn("b: no transcript for 1 of 3", refused.stdout)
            common = self.run_score(d, "--common", "--manifest", "m.jsonl", "a=a.json", "b=b.json")
            self.assertEqual(0, common.returncode, common.stderr)
            self.assertIn("all: 2 utterances, 4 tokens", common.stdout)
            whole = self.run_score(d, "--manifest", "m.jsonl", "a=a.json")
            self.assertIn("all: 3 utterances, 6 tokens", whole.stdout)


if __name__ == "__main__":
    unittest.main()
