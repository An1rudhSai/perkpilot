# perkpilot eval results

> **Offline / scripted mode.** Every number below was produced by `ScriptedLLM`, a deterministic set of
> regex and keyword heuristics standing in for a language model (no API key, no network). The terms text,
> the heuristics and the gold set were written by the same author, so these numbers show that the
> pipeline works end to end and how the three setups differ in *what information they can reach*.
> They are **not** a measurement of any real LLM. Run with `PERKPILOT_LLM_API_KEY` set to get real numbers.

LLM: `offline-scripted` · Q&A items: 40 · extraction checks: 60

## Q&A: three setups

| Metric | Plain LLM | RAG only | Full agent |
|---|---|---|---|
| Answer correctness (answerable) | 47% | 33% | 93% |
| Overall accuracy (incl. abstain items) | 35% | 48% | 95% |
| Citation accuracy (answered, answerable) | 0% | 93% | 100% |
| Abstention rate on unanswerable (higher is better) | 0% | 90% | 100% |
| False abstention on answerable (lower is better) | 0% | 53% | 0% |
| Hallucination rate (lower is better) | 65% | 33% | 7% |

Correct answers by question category:

| Category | n | Plain | RAG | Agent |
|---|---|---|---|---|
| credit_remaining | 4 | 1 | 0 | 4 |
| expiring | 5 | 0 | 0 | 5 |
| fact | 7 | 4 | 6 | 6 |
| fee_worth | 4 | 3 | 0 | 4 |
| unanswerable | 10 | 0 | 9 | 10 |
| which_card | 10 | 6 | 4 | 9 |

Definitions: *answer correctness* = correct machine-readable value on answerable items; *citation accuracy* =
share of answered, answerable items citing at least one gold supporting clause; *hallucination rate* = share of
non-abstained answers that are wrong or answer an unanswerable question.

## Extraction: precision / recall per field

| Field | n | Correct | Wrong | Missing | Spurious | Precision | Recall |
|---|---|---|---|---|---|---|---|
| amount | 8 | 8 | 0 | 0 | 0 | 100% | 100% |
| annual_fee | 4 | 4 | 0 | 0 | 0 | 100% | 100% |
| booking_channel | 8 | 7 | 1 | 0 | 0 | 88% | 88% |
| cap.after_multiplier | 1 | 1 | 0 | 0 | 0 | 100% | 100% |
| cap.amount | 3 | 3 | 0 | 0 | 0 | 100% | 100% |
| eligible_merchants | 2 | 2 | 0 | 0 | 0 | 100% | 100% |
| enrollment_required | 6 | 6 | 0 | 0 | 0 | 100% | 100% |
| exclusions | 4 | 4 | 0 | 0 | 0 | 100% | 100% |
| foreign_transaction_fee_pct | 4 | 4 | 0 | 0 | 0 | 100% | 100% |
| multiplier | 13 | 13 | 0 | 0 | 0 | 100% | 100% |
| period | 5 | 5 | 0 | 0 | 0 | 100% | 100% |
| reset | 2 | 2 | 0 | 0 | 0 | 100% | 100% |
| **all** | 60 | 59 | 1 | 0 | 0 | **98%** | **98%** |

Extraction check failures:

- `x31` amex_platinum hotels.booking_channel: expected `['amex_travel']`, got `['amex_travel', 'hotel_direct']`

Fields rejected for lack of a supporting span: 0

Validator disagreements (logged for review, not auto-fixed):

- `amex_platinum.hotels`: booking channel hotel_direct comes from a sentence with a different rate (1 point per dollar)
- `amex_platinum.uber_cash`: clause also mentions $20, which the rule does not capture

## Agent errors

- `q10` (which_card): expected `['sapphire_reserve']`, got `amex_platinum`
- `q29` (fact): expected `False`, got `True`
