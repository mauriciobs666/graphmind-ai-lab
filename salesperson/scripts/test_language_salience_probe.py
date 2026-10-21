"""Offline tests for language_salience_probe.py — no LM Studio, no server.

Run from the repo root (stdlib unittest; pytest also collects it):

    python3 -m unittest salesperson/scripts/test_language_salience_probe.py -v
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import language_salience_probe as probe  # noqa: E402
import load_demo  # noqa: E402

# Verbatim from falkor-chat/docs/test-reports/salesperson-language-salience-report.md:
# the genuinely-Portuguese reply the v1 classifier misread as Spanish.
REPORT_PT_REPLY = (
    "Por favor, informe o endereço de entrega completo para que eu possa "
    "processar seu pedido corretamente"
)


class ClassifierTests(unittest.TestCase):
    def test_v1_defect_regression_shared_fillers_do_not_outvote_portuguese(self):
        self.assertEqual(probe.classify_language(REPORT_PT_REPLY), "pt-BR")
        self.assertFalse(probe.is_wrong_language("pt-BR", REPORT_PT_REPLY))

    def test_shared_ibero_words_alone_carry_no_signal(self):
        text = "Por favor, que para de entrega pedido está"
        self.assertEqual(probe.language_scores(text), {"en": 0, "pt-BR": 0, "es": 0})
        self.assertEqual(probe.classify_language(text), probe.UNKNOWN)
        self.assertFalse(probe.is_wrong_language("pt-BR", text))
        self.assertFalse(probe.is_wrong_language("es", text))

    def test_portuguese_diacritic_beats_several_shared_words(self):
        text = "Por favor, que para de a o se no — confirmação"
        self.assertEqual(probe.classify_language(text), "pt-BR")

    def test_exclusive_diacritic_alone_is_a_signal(self):
        # No lexicon word and no suffix: only the orthography decides.
        self.assertEqual(probe.classify_language("Açúcar mascavo"), "pt-BR")
        self.assertEqual(probe.classify_language("¿Algo?"), "es")
        # one exclusive diacritic outweighs one exclusive word of the other language
        self.assertEqual(probe.classify_language("Açúcar y"), "pt-BR")

    def test_clear_cases(self):
        self.assertEqual(probe.classify_language("Hello! How can I help you today?"), "en")
        self.assertEqual(
            probe.classify_language("¡Hola! ¿En qué puedo ayudarte hoy? Tenemos muchos productos."), "es")
        self.assertEqual(
            probe.classify_language("Olá! Como posso ajudar você hoje? Temos vários produtos."), "pt-BR")
        self.assertEqual(
            probe.classify_language("Por favor, indícame tu dirección de entrega para completar el pedido."),
            "es")

    def test_k065_shape_spanish_reply_to_english_participant_is_wrong(self):
        reply = "¡Claro! He añadido el Wireless Mouse Pro a tu carrito. ¿Deseas algo más?"
        self.assertEqual(probe.classify_language(reply), "es")
        self.assertTrue(probe.is_wrong_language("en", reply))
        self.assertTrue(probe.is_wrong_language("pt-BR", reply))
        self.assertFalse(probe.is_wrong_language("es", reply))

    def test_english_product_name_inside_portuguese_reply_stays_portuguese(self):
        reply = "Adicionei o Wireless Mouse Pro ao seu carrinho. Deseja mais alguma coisa?"
        self.assertEqual(probe.classify_language(reply), "pt-BR")
        self.assertFalse(probe.is_wrong_language("pt-BR", reply))

    def test_no_reply_is_unknown_never_wrong(self):
        for text in (None, "", "   "):
            self.assertEqual(probe.classify_language(text), probe.UNKNOWN)
            self.assertFalse(probe.is_wrong_language("en", text))

    def test_tie_is_unknown(self):
        self.assertEqual(probe.classify_language("eu yo"), probe.UNKNOWN)


class WilsonTests(unittest.TestCase):
    def assertCI(self, x, n, lo_pct, hi_pct):
        lo, hi = probe.wilson_interval(x, n)
        self.assertAlmostEqual(lo * 100, lo_pct, places=1, msg=f"low for {x}/{n}")
        self.assertAlmostEqual(hi * 100, hi_pct, places=1, msg=f"high for {x}/{n}")

    def test_figures_published_in_the_2026_09_18_report(self):
        self.assertCI(61, 61, 94.1, 100.0)
        self.assertCI(46, 46, 92.3, 100.0)
        self.assertCI(33, 33, 89.6, 100.0)
        self.assertCI(0, 23, 0.0, 14.3)
        self.assertCI(2, 10, 5.7, 51.0)
        self.assertCI(1, 5, 3.6, 62.4)

    def test_interior_values_against_quadratic_roots(self):
        # Independent derivation: the Wilson bounds are the roots p of
        # (p_hat - p)^2 = z^2 p (1 - p) / n, i.e.
        # (1 + z^2/n) p^2 - (2 p_hat + z^2/n) p + p_hat^2 = 0.
        z = 1.959963984540054
        for x, n in ((7, 40), (1, 3), (12, 140), (2, 10)):
            ph = x / n
            a, b, c = 1 + z * z / n, -(2 * ph + z * z / n), ph * ph
            disc = (b * b - 4 * a * c) ** 0.5
            lo, hi = probe.wilson_interval(x, n)
            self.assertAlmostEqual(lo, (-b - disc) / (2 * a), places=9)
            self.assertAlmostEqual(hi, (-b + disc) / (2 * a), places=9)

    def test_symmetry_and_bounds(self):
        for n in (1, 5, 23, 140):
            for x in range(n + 1):
                lo, hi = probe.wilson_interval(x, n)
                lo2, hi2 = probe.wilson_interval(n - x, n)
                self.assertTrue(0.0 <= lo <= x / n + 1e-12 and x / n - 1e-12 <= hi <= 1.0)
                self.assertAlmostEqual(lo, 1 - hi2, places=9)
                self.assertAlmostEqual(hi, 1 - lo2, places=9)

    def test_empty_and_invalid(self):
        self.assertEqual(probe.wilson_interval(0, 0), (0.0, 1.0))
        with self.assertRaises(ValueError):
            probe.wilson_interval(3, 2)


def _turn(trial, lang, idx, reply, dead=False, post_status=200, classified_at_run=None):
    return {
        "type": "turn", "trial": trial, "language": lang, "turn_index": idx,
        "reply_text": reply, "dead_turn": dead, "post_status": post_status,
        "timed_out": False, "classified_at_run": classified_at_run,
    }


EN_OK = "Sure, I can help you with that."
ES_REPLY = "¡Claro! ¿Qué productos quieres?"
PT_OK = "Claro, posso ajudar você com isso."


class AggregateTests(unittest.TestCase):
    def records(self):
        return [
            {"type": "run_start", "run_id": "r"},
            # trial 0: en 2 replied (1 dead-but-replied) + 1 dead no-reply; pt 1 replied + 1 silent no-reply
            _turn(0, "en", 0, EN_OK),
            _turn(0, "en", 1, EN_OK, dead=True),
            _turn(0, "en", 2, None, dead=True),
            _turn(0, "pt-BR", 0, PT_OK),
            _turn(0, "pt-BR", 1, None, dead=False),
            {"type": "trial_end", "trial": 0},
            # trial 1: en gets one Spanish reply (affected); pt all dead
            # classified_at_run deliberately lies — aggregate must re-classify from text
            _turn(1, "en", 0, ES_REPLY, classified_at_run="en"),
            _turn(1, "en", 1, EN_OK, classified_at_run="es"),
            _turn(1, "pt-BR", 0, None, dead=True),
            _turn(1, "pt-BR", 1, None, dead=True, post_status=503),
            {"type": "trial_end", "trial": 1},
            # trial 2: killed mid-trial — no trial_end
            _turn(2, "en", 0, ES_REPLY),
        ]

    def test_turn_level_counts_replied_versus_dead(self):
        agg = probe.aggregate(self.records())
        en = agg["per_language"]["en"]
        self.assertEqual(en["turns"], 5)
        self.assertEqual(en["replied"], 4)
        self.assertEqual(en["wrong"], 1)
        self.assertEqual(en["adherent"], 3)
        self.assertEqual(en["no_reply"], 1)
        self.assertEqual(en["dead_but_replied"], 1)
        self.assertEqual(en["dead_total"], 2)
        self.assertEqual(en["no_reply_not_dead"], 0)
        pt = agg["per_language"]["pt-BR"]
        self.assertEqual((pt["replied"], pt["no_reply"], pt["dead_total"], pt["no_reply_not_dead"]), (1, 3, 2, 1))
        self.assertEqual(pt["post_errors"], 1)
        self.assertEqual(agg["totals"],
                         {"turns": 9, "no_reply": 4, "dead_but_replied": 1, "dead_total": 4, "no_reply_not_dead": 1})

    def test_trial_level_denominator_is_trials_with_a_reply(self):
        agg = probe.aggregate(self.records())
        en, pt = agg["per_language"]["en"], agg["per_language"]["pt-BR"]
        self.assertEqual((en["trials"], en["trials_with_reply"], en["trials_affected"]), (2, 2, 1))
        self.assertEqual(en["affected_trial_ids"], [1])
        self.assertEqual((pt["trials"], pt["trials_with_reply"], pt["trials_affected"]), (2, 1, 0))

    def test_partial_trial_excluded_unless_asked(self):
        agg = probe.aggregate(self.records())
        self.assertEqual(agg["trials_complete"], 2)
        self.assertEqual(agg["turn_records_excluded_partial"], 1)
        agg2 = probe.aggregate(self.records(), include_partial=True)
        self.assertEqual(agg2["per_language"]["en"]["wrong"], 2)
        self.assertEqual(agg2["per_language"]["en"]["trials_affected"], 2)

    def test_per_trial_cells(self):
        agg = probe.aggregate(self.records())
        self.assertEqual(agg["per_trial"][0]["en"], {"adherent": 2, "replied": 2, "dead": 2, "wrong": 0})
        self.assertEqual(agg["per_trial"][1]["en"], {"adherent": 1, "replied": 2, "dead": 0, "wrong": 1})
        self.assertEqual(agg["per_trial"][1]["pt-BR"], {"adherent": 0, "replied": 0, "dead": 2, "wrong": 0})

    def test_render_smoke(self):
        recs = self.records()
        text = probe.render(probe.aggregate(recs), recs, dump_replies=True, include_partial=False)
        self.assertIn("| `en` | 4 | 3 | 1 | 0 |", text)
        self.assertIn("WRONG es", text)


class FileAndParityTests(unittest.TestCase):
    def test_scripted_turns_match_load_demo(self):
        self.assertEqual(probe.SCRIPTED_TURNS, load_demo.SCRIPTED_TURNS)
        self.assertEqual(probe.DEFAULT_LANGUAGES, load_demo.LANGUAGES)

    def test_writer_is_line_per_record_and_torn_tail_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.jsonl")
            w = probe.JsonlWriter(path)
            w.write({"type": "turn", "trial": 0})
            w.write({"type": "trial_end", "trial": 3})
            w.close()
            with open(path, "a") as f:
                f.write('{"type": "turn", "tri')  # killed mid-write
            recs = probe.read_records(path)
            self.assertEqual([r["type"] for r in recs], ["turn", "trial_end"])
            self.assertEqual(probe.next_trial_index(path), 4)
            self.assertEqual(probe.next_trial_index(os.path.join(d, "absent.jsonl")), 0)

    def test_run_refuses_to_clobber_without_resume(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.jsonl")
            with open(path, "w") as f:
                f.write(json.dumps({"type": "run_start"}) + "\n")
            rc = probe.main(["run", "--trials", "1", "--out", path, "--base-url", "http://127.0.0.1:9"])
            self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
