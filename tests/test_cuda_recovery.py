import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch is not installed")
class CudaRecoveryTest(unittest.TestCase):
    def test_oom_batch_is_recursively_split(self):
        import torch

        from molbench.core.batching import PreparedInput
        from molbench.core.model import (
            GenerationBatch,
            GenerationConfig,
            GenerationInput,
            GenerationOutput,
        )
        from molbench.models.chemdfm import ChemDFMModel

        model = object.__new__(ChemDFMModel)

        def decode(batch, config, batch_id):
            if len(batch) > 1:
                raise torch.cuda.OutOfMemoryError("injected out of memory")
            item = batch[0]
            return GenerationBatch(
                batch_id=batch_id,
                outputs=[
                    GenerationOutput(
                        item.item.example_index,
                        "ok",
                        "ok",
                        item.prompt_tokens,
                        1,
                        "eos",
                        item.item.size_hint,
                    )
                ],
                elapsed_seconds=0,
                remaining_examples=0,
            )

        model._decode_batch = decode
        items = [
            PreparedInput(GenerationInput(i, str(i), str(i), 1), str(i), 1)
            for i in range(4)
        ]
        batches = list(
            model._iter_with_oom_split(items, GenerationConfig(), next_id=[1])
        )
        self.assertEqual([b.outputs[0].example_index for b in batches], [0, 1, 2, 3])
        self.assertTrue(all(len(b.outputs) == 1 for b in batches))


if __name__ == "__main__":
    unittest.main()
