"""
Orca-Math word problems, as an RL pool.
https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k

200,035 grade-school word problems. Same shape as GSM8K but ~27x the problems,
which is the point: the GSM8K train split is also what our distilled SFT
checkpoints were trained on, so RL over it mostly buys memorization. This pool
is unseen by those checkpoints.

The source has no `#### <answer>` marker -- `answer` is a free-form worked
solution and the final number is just the last thing said. We therefore KEEP
ONLY problems whose gold answer extracts unambiguously and append the marker
ourselves, so grading is byte-identical to GSM8K's (this module reuses its
`evaluate`/`reward` directly rather than reimplementing them).

Filter (see `_gold`), measured on a 6,000-row sample:
  - last non-empty line of the solution holds EXACTLY ONE number  -> 63.1% kept.
    Multi-number last lines are usually multi-part questions ("find the second
    and third smallest...") which have no single numeric answer at all.
  - that number must be an INTEGER -> 51.7% of all rows kept (~103k problems).
    Decimal golds ("approximately 116.69") are real answers but make a
    string-equality reward brittle, and our SFT substrate was distilled on
    integer-answer GSM8K traces.
  - question length capped so the prefill pack stays bounded (p95 is 113
    tokens, p99 188, max 417 -- the cap only trims the far tail).
Spot-checked 12/12 correct against the questions; where the solution also
carries an explicit "the answer is X" phrase (17.9% of rows) it agrees with
the extracted number 100% of the time.
"""

import re

from tasks.common import Task, load_hub_dataset
from tasks.gsm8k import GSM8K, extract_answer  # noqa: F401  (extract_answer re-exported)


NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
INT_RE = re.compile(r"-?\d+")


def _gold(answer):
    """The unambiguous integer answer of a solution, or None to drop the row."""
    lines = [ln for ln in answer.strip().split("\n") if ln.strip()]
    if not lines:
        return None
    nums = NUM_RE.findall(lines[-1])
    if len(nums) != 1:
        return None
    # "the value of A is 4." -> the regex takes the sentence period with it
    value = nums[0].replace(",", "").rstrip(".")
    return value if INT_RE.fullmatch(value) else None


class OrcaMath(Task):

    # Grading is GSM8K's, unmodified: extract `#### n` from both the reference
    # and the completion and compare. Bound here explicitly so the two pools
    # can never drift apart.
    evaluate = GSM8K.evaluate
    reward = GSM8K.reward

    def __init__(self, split="train", max_question_chars=1400, **kwargs):
        super().__init__(**kwargs)
        assert split == "train", "orca-math ships a single split"
        self.ds = load_hub_dataset(
            "microsoft/orca-math-word-problems-200k", split=split).shuffle(seed=42)
        # Filter once at construction; `keep` indexes the shuffled dataset and
        # `golds` is parallel to it, so nothing is re-extracted per access.
        table = self.ds.table
        questions = table["question"].to_pylist()
        answers = table["answer"].to_pylist()
        perm = self.ds.permutation
        self.keep, self.golds = [], []
        for i in range(len(self.ds)):
            phys = i if perm is None else int(perm[i])
            if len(questions[phys]) > max_question_chars:
                continue
            g = _gold(answers[phys])
            if g is None:
                continue
            self.keep.append(i)
            self.golds.append(g)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.keep)

    def get_example(self, index):
        row = self.ds[self.keep[index]]
        # Append the marker the reward reads. The worked solution is kept ahead
        # of it so this task is also usable as SFT text, exactly like GSM8K's.
        answer = f"{row['answer'].strip()}\n#### {self.golds[index]}"
        messages = [
            {"role": "user", "content": row['question']},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        return {"messages": messages}
