import unittest

from molbench.core.model import GenerationConfig, GenerationInput
from molbench.core.batching import (
    PreparedInput,
    length_band,
    parse_length_batch_policy,
    plan_batches,
)


class BatchingTest(unittest.TestCase):
    def test_policy_matches_chebi_long_tail_rules(self):
        bands = parse_length_batch_policy("128:16,256:8,384:4,512:2,inf:1")
        self.assertEqual(length_band(52, bands)[1], 16)
        self.assertEqual(length_band(193, bands)[1], 8)
        self.assertEqual(length_band(384, bands)[1], 4)
        self.assertEqual(length_band(496, bands)[1], 2)
        for length in (742, 808, 1154):
            self.assertEqual(length_band(length, bands)[1], 1)

    def test_longest_bands_first_and_token_budget_enforced(self):
        lengths = [52, 60, 742, 70, 300, 808, 1154]
        prepared = [
            PreparedInput(
                item=GenerationInput(i, str(i), str(i), length),
                chat_prompt=str(i),
                prompt_tokens=max(10, length // 2),
            )
            for i, length in enumerate(lengths)
        ]
        config = GenerationConfig(
            max_new_tokens=512,
            max_batch_size=16,
            token_budget=16384,
        )
        planned = plan_batches(prepared, config)
        flattened = [item.item.size_hint for _, batch in planned for item in batch]
        self.assertEqual(flattened[:3], [1154, 808, 742])
        self.assertTrue(all(len(batch) == 1 for _, batch in planned[:3]))
        for _, batch in planned:
            max_prompt = max(item.prompt_tokens for item in batch)
            self.assertLessEqual(len(batch) * (max_prompt + 512), 16384)

    def test_fixed_batching_also_honors_token_budget(self):
        token_lengths = [10, 60, 60]
        prepared = [
            PreparedInput(
                item=GenerationInput(i, str(i), str(i), 10),
                chat_prompt=str(i),
                prompt_tokens=tokens,
            )
            for i, tokens in enumerate(token_lengths)
        ]
        config = GenerationConfig(
            max_new_tokens=20,
            max_batch_size=3,
            batching="fixed",
            token_budget=160,
            max_padding_ratio=100,
        )
        planned = plan_batches(prepared, config)
        indexes = [[item.item.example_index for item in batch] for _, batch in planned]
        self.assertEqual(indexes, [[0, 1], [2]])
        for _, batch in planned:
            max_prompt = max(item.prompt_tokens for item in batch)
            self.assertLessEqual(len(batch) * (max_prompt + 20), 160)

    def test_single_example_over_token_budget_is_rejected(self):
        prepared = [
            PreparedInput(
                item=GenerationInput(7, "7", "7", 10),
                chat_prompt="7",
                prompt_tokens=200,
            )
        ]
        config = GenerationConfig(max_new_tokens=20, token_budget=128)
        with self.assertRaisesRegex(ValueError, "batch size 1"):
            plan_batches(prepared, config)

    def test_long_prompt_is_always_isolated_by_real_token_length(self):
        prepared = [
            PreparedInput(
                item=GenerationInput(i, str(i), str(i), 20),
                chat_prompt=str(i),
                prompt_tokens=tokens,
            )
            for i, tokens in enumerate((100, 1100, 105))
        ]
        config = GenerationConfig(
            max_new_tokens=100,
            max_batch_size=8,
            token_budget=4096,
            long_prompt_threshold=1024,
        )
        planned = plan_batches(prepared, config)
        batches = [[item.item.example_index for item in batch] for _, batch in planned]
        self.assertIn([1], batches)
        self.assertFalse(any(1 in batch and len(batch) > 1 for batch in batches))

    def test_padding_ratio_prevents_short_and_long_prompt_mixing(self):
        prepared = [
            PreparedInput(
                item=GenerationInput(i, str(i), str(i), 20),
                chat_prompt=str(i),
                prompt_tokens=tokens,
            )
            for i, tokens in enumerate((100, 120, 220))
        ]
        config = GenerationConfig(
            max_new_tokens=64,
            max_batch_size=8,
            token_budget=4096,
            max_padding_ratio=1.25,
        )
        planned = plan_batches(prepared, config)
        batches = [[item.item.example_index for item in batch] for _, batch in planned]
        self.assertEqual(batches, [[2], [1, 0]])


if __name__ == "__main__":
    unittest.main()
