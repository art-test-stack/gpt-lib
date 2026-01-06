import pytest
import torch
from gpt_lib.model.model import GPTModel
from gpt_lib.utils.schemas import (
    GPTConfig, 
    ObjectiveConfig, 
    TokenizerConfig, 
    TransformerConfig,
    GenerationConfig,
    ModelOutput,
)
import tempfile

class TestGPTModel:
    model_name = "test-model"
    pad_token_id = 0
    tmpdirname = tempfile.mkdtemp()
    tokenizer_config = TokenizerConfig(
        vocab_size=1000,
        max_context=16,
        name="simple-tokenizer",
        source="dummy"
    )
    model_config = TransformerConfig(
        vocab_size=1000,
        pad_id=pad_token_id,
        max_context=16,
        d_model=16,
        d_ffn=64,
        n_heads=4,
        n_layers=4,
        d_head=4,
        dropout=0.1
    )
    obj_config = ObjectiveConfig(
        loss_fn="cross_entropy",
        ignore_index=pad_token_id,
        kwargs={"reduction": "mean"}
    )
    config = GPTConfig(
        name=model_name,
        tokenizer=tokenizer_config,
        model=model_config,
        objective=obj_config,
        dirname=tmpdirname
    )

    # TESTS

    @pytest.mark.fast
    def test_model_loading_saving(self):
        self.config.to_file(mode="pickle")

        loaded_config = GPTConfig.from_file(model_name=self.model_name, model_dir=self.tmpdirname)

        for key, value in self.config.__dict__.items():
            assert key in loaded_config.__dict__, f"Key {key} missing in loaded config"
            assert getattr(loaded_config, key) is not None, f"Key {key} is None in loaded config"
            assert getattr(loaded_config, key) == value, f"Value for key {key} does not match: {getattr(loaded_config, key)} != {value}"
        assert loaded_config == self.config, "Loaded config does not match the original"
        model = GPTModel.from_scratch(config=self.config)

        model.save_checkpoint(ckpt_version="test-1", keep_vars=True)

        loaded_model = GPTModel.load(model_name=self.model_name, ckpt_version="test-1", model_dir=self.tmpdirname)
        assert loaded_model.config == self.config, "Loaded model config does not match the original"
        assert loaded_model.model.state_dict().keys() == model.model.state_dict().keys(), "Loaded model state dict keys do not match the original"
        assert all(torch.equal(loaded_model.model.state_dict()[k], model.model.state_dict()[k]) for k in model.model.state_dict().keys()), "Loaded model state dict values do not match the original"


    @pytest.mark.fast
    def test_model_generation(self):
        config = self.config

        model = GPTModel.from_scratch(config)
        max_context = config.model.max_context
        vocab_size = config.model.vocab_size
        batch_size = 4
        # Dummy input ids and labels
        input_ids = torch.randint(0, vocab_size, (batch_size, max_context))
        labels = torch.randint(0, vocab_size, (batch_size, max_context))

        generation_config = GenerationConfig(
            max_length=20,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.0,
            do_sample=True,
            num_return_sequences=1,
            stream=False
        )
        with torch.no_grad():
            model.eval()
            output = model.forward(input_ids=input_ids, labels=labels, **generation_config.__dict__)
            logits = model(
                input_ids=input_ids,
                return_attentions=False,
                # log_prob=False,
                # temperature=generation_config.temperature
            ).logits
        assert isinstance(output, ModelOutput), "Output is not an instance of ModelCompletionOutput"
        assert logits.size(0) == batch_size, "Logits batch size does not match input batch size"
        assert logits.size(1) == max_context, "Logits sequence length does not match input sequence length"
        assert logits.size(2) == vocab_size, "Logits vocab size does not match model vocab size"
        assert logits.nansum() != 0, "Logits contain NaN values"
        assert torch.isfinite(logits).all(), "Logits contain non-finite values"
        assert (logits == output.logits).all(), "Logits from forward method do not match logits from __call__ method"
