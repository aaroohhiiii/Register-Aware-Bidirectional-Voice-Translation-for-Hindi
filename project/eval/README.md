# Evaluation & Ratings Sheet

## Evaluation Dimensions
1. **ASR Word Error Rate (WER)**: Transcribing speech accurately across varying accents.
2. **Filler Removal Precision**: Removing hesitation tokens while preserving core utterance tokens.
3. **Register Transformation Accuracy**: Correct pronoun paradigm, verb agreement, honorific particles, and vocatives.
4. **Naturalness (MOS 1-5)**: Perceived naturalness of transformed Hindi speech.

## Test Sets
- `register_test_sentences.json`: 48 canonical sentences across 6 tiers.
- `cross_tier_review.py`: 144 transformation pairs across registers.
