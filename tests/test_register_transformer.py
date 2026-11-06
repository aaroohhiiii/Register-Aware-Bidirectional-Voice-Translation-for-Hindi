import unittest

from cross_tier_review import TARGET_SCHEDULE, build_review_cases
from hindi_tokenizer import tokenize_and_tag
from register_transformer import transform


class RegisterTransformerTests(unittest.TestCase):
    def test_accepts_raw_text_and_tagged_tokens(self) -> None:
        sentence = "तू क्या कर रही है?"
        expected = "आप क्या कर रही हैं?"
        self.assertEqual(transform(sentence, "Formal", "F"), expected)
        self.assertEqual(
            transform(tokenize_and_tag(sentence), "Formal", "feminine"), expected
        )

    def test_all_pronoun_cases_and_possessive_genders(self) -> None:
        cases = {
            "तू आज आएगा": "आप आज आएगा",
            "तूने खाना खाया": "आपने खाना खाया",
            "तुझे पता है": "आपको पता है",
            "तेरा बैग": "आपका बैग",
            "तेरी रसीद": "आपकी रसीद",
            "तेरे पैसे": "आपके पैसे",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(transform(source, "Formal"), expected)

    def test_verb_agreement_by_tier_gender_and_tense(self) -> None:
        cases = (
            ("तू क्या कर रहा है?", "Respectful Casual", "M", "तुम क्या कर रहे हो?"),
            ("तू क्या कर रही है?", "Formal", "F", "आप क्या कर रही हैं?"),
            ("तुम कहाँ रहते हो?", "Full Informal", "M", "तू कहाँ रहता है?"),
            ("आप ये कर सकते हैं?", "Full Informal", "M", "तू ये कर सकता है?"),
            ("तुम काम कर चुकी हो", "Formal", "F", "आप काम कर चुकी हैं"),
            ("तुम कल आओगी?", "Formal", "F", "आप कल आएंगी?"),
        )
        for source, tier, gender, expected in cases:
            with self.subTest(source=source, tier=tier):
                self.assertEqual(transform(source, tier, gender), expected)

    def test_honorific_imperative_and_address_policies(self) -> None:
        self.assertEqual(
            transform("आप कृपया दोबारा समझाइए", "Full Informal"),
            "तू दोबारा समझा",
        )
        self.assertEqual(
            transform("तुम जल्दी आ जाना", "Formal"),
            "आप कृपया जल्दी आ जाइए",
        )
        self.assertEqual(
            transform("तू क्या कर रहा है?", "Full Formal", "M"),
            "सर, आप क्या कर रहे हैं?",
        )
        self.assertEqual(
            transform("भैया, तुम्हारे पास पैसे हैं?", "Reverse Formal", "M"),
            "सर, आपके पास पैसे हैं?",
        )

    def test_formal_warm_preserves_real_kinship_but_never_guesses(self) -> None:
        self.assertEqual(
            transform("दादी, तुम कैसी हो?", "Formal-Warm", "F"),
            "दादी जी, आप कैसी हैं?",
        )
        self.assertEqual(
            transform("तुम कैसी हो?", "Formal-Warm", "F"),
            "आप कैसी हैं?",
        )

    def test_first_person_and_object_agreement_are_not_rewritten(self) -> None:
        self.assertEqual(
            transform("सर, क्या मैं अंदर आ सकता हूं?", "Full Informal"),
            "भाई, क्या मैं अंदर आ सकता हूं?",
        )
        self.assertEqual(
            transform("तूने रोटी खा ली?", "Formal", "F"),
            "आपने रोटी खा ली?",
        )

    def test_imperative_agrees_with_converted_pronoun(self) -> None:
        self.assertEqual(
            transform(
                "आप अपना बैग पीछे रखिए मैं आपको जल्दी स्टेशन तक पहुंचा दूंगा।",
                "Respectful Casual",
            ),
            "तुम अपना बैग पीछे रखो मैं तुम्हें जल्दी स्टेशन तक पहुंचा दूंगा।",
        )

    def test_extended_imperatives_convert_across_tiers(self) -> None:
        cases = {
            "रखिए": ("रख", "रखो"),
            "रुकिए": ("रुक", "रुको"),
            "लीजिए": ("ले", "लो"),
            "बोलिए": ("बोल", "बोलो"),
            "सुनिए": ("सुन", "सुनो"),
            "देखिए": ("देख", "देखो"),
            "चलिए": ("चल", "चलो"),
            "आइए": ("आ", "आओ"),
            "जाइए": ("जा", "जाओ"),
            "पढ़िए": ("पढ़", "पढ़ो"),
            "लिखिए": ("लिख", "लिखो"),
            "दिखाइए": ("दिखा", "दिखाओ"),
            "बताइए": ("बता", "बताओ"),
            "लाइए": ("ला", "लाओ"),
            "खाइए": ("खा", "खाओ"),
            "पीजिए": ("पी", "पियो"),
            "रहिए": ("रह", "रहो"),
            "उठिए": ("उठ", "उठो"),
            "भेजिए": ("भेज", "भेजो"),
            "मिलिए": ("मिल", "मिलो"),
        }
        for formal, (intimate, casual) in cases.items():
            # जाइए is the one verb here that _has_imperative lists, so it
            # also gains कृपया when converted up to Formal.
            formal_output = "आप कृपया जाइए" if formal == "जाइए" else f"आप {formal}"
            with self.subTest(formal=formal):
                self.assertEqual(transform(f"आप {formal}", "Full Informal"), f"तू {intimate}")
                self.assertEqual(transform(f"आप {formal}", "Respectful Casual"), f"तुम {casual}")
                self.assertEqual(transform(f"तुम {casual}", "Formal"), formal_output)
                self.assertEqual(transform(f"तू {intimate}", "Formal"), formal_output)

    def test_short_stems_in_verb_phrases_are_not_imperatives(self) -> None:
        cases = (
            ("तू खाना खा रहा है", "Formal", "M", "आप खाना खा रहे हैं"),
            ("तू बैग रख दे", "Respectful Casual", "M", "तुम बैग रख दो"),
            ("तू चल सकता है", "Formal", "M", "आप चल सकते हैं"),
            ("मैं जा रहा हूं", "Formal", "M", "मैं जा रहा हूं"),
        )
        for source, tier, gender, expected in cases:
            with self.subTest(source=source, tier=tier):
                self.assertEqual(transform(source, tier, gender), expected)

    def test_unknown_forms_follow_fallback_rule(self) -> None:
        # Lexical items outside the tables still follow graceful fallback.
        self.assertEqual(
            transform("तुम ज़ाइलोफोन बजाओ", "Formal"),
            "आप ज़ाइलोफोन बजाओ",
        )

    def test_instrumental_gap_is_closed_across_all_tiers(self) -> None:
        expected = {
            "Full Informal": "तुझसे मिलकर खुशी हुई",
            "Respectful Casual": "तुमसे मिलकर खुशी हुई",
            "Formal": "आपसे मिलकर खुशी हुई",
            "Full Formal": "जी, आपसे मिलकर खुशी हुई",
            "Formal-Warm": "आपसे मिलकर खुशी हुई",
            "Reverse Formal": "जी, आपसे मिलकर खुशी हुई",
        }
        for tier, output in expected.items():
            with self.subTest(tier=tier):
                self.assertEqual(transform("आपसे मिलकर खुशी हुई", tier), output)

    def test_audited_postpositional_cases_and_spelling_variants(self) -> None:
        cases = (
            ("आपमें धैर्य है", "Full Informal", "तुझमें धैर्य है"),
            ("आप में धैर्य है", "Respectful Casual", "तुममें धैर्य है"),
            ("तुम पर भरोसा है", "Formal", "आप पर भरोसा है"),
            ("तुमपे भरोसा है", "Full Informal", "तुझ पर भरोसा है"),
            ("तुझ तक खबर पहुँची", "Formal", "आप तक खबर पहुँची"),
            ("तेरे साथ चलूँगा", "Formal", "आपके साथ चलूँगा"),
            ("तुम्हारे लिए चाय है", "Full Informal", "तेरे लिए चाय है"),
            ("आपके पास समय है", "Respectful Casual", "तुम्हारे पास समय है"),
        )
        for source, tier, expected in cases:
            with self.subTest(source=source, tier=tier):
                self.assertEqual(transform(source, tier), expected)

    def test_invalid_arguments_fail_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown target tier"):
            transform("तुम ठीक हो", "Ultra Formal")
        with self.assertRaisesRegex(ValueError, "gender must be"):
            transform("तुम ठीक हो", "Formal", "unknown")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            transform("", "Formal")

    def test_cross_tier_matrix_covers_three_targets_for_all_48(self) -> None:
        cases = build_review_cases()
        self.assertEqual(len(cases), 144)
        self.assertEqual({case["sentence_id"] for case in cases}, set(range(1, 49)))
        for source_tier, targets in TARGET_SCHEDULE.items():
            self.assertEqual(len(targets), 3)
            self.assertNotIn(source_tier, targets)

        counts = {sentence_id: 0 for sentence_id in range(1, 49)}
        for case in cases:
            counts[case["sentence_id"]] += 1
            self.assertTrue(case["output"].strip())
            self.assertFalse(case["output"].startswith(","))
        self.assertEqual(set(counts.values()), {3})

    def test_cross_tier_low_registers_remove_politeness_particles(self) -> None:
        for case in build_review_cases():
            if case["target_tier"] not in {"Full Informal", "Respectful Casual"}:
                continue
            tokens = {token.text for token in tokenize_and_tag(case["output"])}
            self.assertNotIn("जी", tokens, case)
            self.assertNotIn("कृपया", tokens, case)


if __name__ == "__main__":
    unittest.main()
