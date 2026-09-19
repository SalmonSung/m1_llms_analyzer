"""Model service: owns loading, hardware placement, and layer introspection.

This is the only module that talks to `transformers` loading APIs. Keeping it
alone behind `ModelProvider` means a swap (quantized loading, a remote inference
host, a stub in tests) touches one file.

Layer indexing convention
-------------------------
`output_hidden_states=True` returns a tuple of length ``num_hidden_layers + 1``:

    hidden_states[0]  -> embedding output (before block 1)
    hidden_states[i]  -> output of transformer block i, for i in 1..N

So on a 24-block model, ``-1`` and ``24`` are the same tensor, and the "middle"
layer is block ``N // 2``. Both the label you asked for and the resolved absolute
index are written into every output file, so a saved result is never ambiguous.

Heads
-----
``ModelConfig.head`` picks the class: ``"base"`` loads ``AutoModel`` (the bare
transformer, hidden states only) and ``"causal_lm"`` loads
``AutoModelForCausalLM`` (adds the language-model head, so next-token
log-probabilities can be scored). A causal-LM model still returns hidden
states, so extraction works with either; the base head is simply lighter.

Multimodal (vision + text) models
---------------------------------
Composite checkpoints such as Gemma 3/4, Llama-3.2-Vision or Qwen-VL carry a
``vision_config`` next to a ``text_config`` that holds the language model. This
pipeline only ever passes ``input_ids`` / ``attention_mask``, so the wrapper
behaves exactly like its language model: ``hidden_states`` and ``logits`` are the
text stack's. Such models are therefore loaded in text-only mode with a warning,
and layer count / hidden size / context length are read from ``text_config``.
The vision tower's weights are loaded but never run. Encoder-decoder models
(including vision-encoder-decoder ones such as TrOCR) are still rejected.
"""

from __future__ import annotations

import platform
from typing import Any, Sequence

from ..config.settings import ModelConfig
from ..utils.device import describe_device, resolve_device, resolve_dtype
from ..utils.env import redact, resolve_hf_token
from ..utils.logging import get_logger

log = get_logger("model")

#: Fallback when a model config exposes no usable context length.
_DEFAULT_MAX_LENGTH = 512

#: Values transformers uses as "no real limit" sentinels.
_SENTINEL_LENGTHS = {int(1e30), 1000000000000000019884624838656}


class ModelLoadError(RuntimeError):
    """Raised with an actionable message when a model cannot be loaded."""


class UnsupportedArchitectureError(ModelLoadError):
    """Raised for architectures whose hidden states this pipeline cannot read yet."""


class ModelService:
    """Loads a Hugging Face model once and answers questions about it."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None
        self._dtype: Any = None
        self._resolved_revision: str | None = None
        self._pad_token_added = False
        self._multimodal = False

    # ------------------------------------------------------------------ load

    def load(self) -> "ModelService":
        """Load tokenizer + model. Idempotent: a second call is a no-op."""
        if self._model is not None:
            return self

        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

        token = resolve_hf_token(self.config.hf_token)
        log.info(
            "Loading %s (head=%s, revision=%s, hf_token=%s)",
            self.config.model_id,
            self.config.head,
            self.config.revision or "default",
            redact(token),
        )

        common = {
            "revision": self.config.revision,
            "trust_remote_code": self.config.trust_remote_code,
            "cache_dir": self.config.cache_dir,
        }
        # `token=None` is the ordinary ungated path -- transformers treats it as
        # "anonymous", so nothing changes when no secret is configured.
        if token:
            common["token"] = token

        try:
            hf_config = AutoConfig.from_pretrained(self.config.model_id, **common)
        except Exception as exc:  # noqa: BLE001 - re-raised with guidance below
            raise self._explain_load_failure(exc) from exc

        self._reject_unsupported(hf_config)
        if self.config.head == "causal_lm":
            self._require_causal_lm(hf_config)
        loader = AutoModelForCausalLM if self.config.head == "causal_lm" else AutoModel

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, **common)
        except Exception as exc:  # noqa: BLE001
            raise self._explain_load_failure(exc) from exc

        self._device = resolve_device(self.config.device)
        self._dtype = resolve_dtype(self._device, self.config.dtype)

        try:
            self._model = loader.from_pretrained(
                self.config.model_id, config=hf_config, dtype=self._dtype, **common
            )
        except TypeError:
            # transformers < 4.56 spells the argument `torch_dtype`.
            self._model = loader.from_pretrained(
                self.config.model_id, config=hf_config, torch_dtype=self._dtype, **common
            )
        except Exception as exc:  # noqa: BLE001
            raise self._explain_load_failure(exc) from exc

        self._model.to(self._device)
        self._model.eval()

        self._ensure_pad_token()
        self._resolved_revision = self._detect_revision()

        log.info(
            "Loaded %s: %d layers, hidden_size=%d, device=%s, dtype=%s",
            self.config.model_id,
            self.num_hidden_layers,
            self.hidden_size,
            self._device,
            str(self._dtype).replace("torch.", ""),
        )
        return self

    def _ensure_pad_token(self) -> None:
        """GPT-2-family tokenizers have no PAD token; batching needs one.

        Reusing EOS is the standard fix and is safe here because the attention
        mask excludes those positions from both the forward pass and pooling.
        Vocabulary is not resized, so the model's embedding matrix is untouched.
        """
        tok = self._tokenizer
        if tok.pad_token_id is not None:
            return
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
            self._pad_token_added = True
            log.info("Tokenizer has no pad_token; using eos_token (%r) for padding.", tok.eos_token)
            return
        if tok.unk_token is not None:
            tok.pad_token = tok.unk_token
            self._pad_token_added = True
            log.info("Tokenizer has no pad_token/eos_token; using unk_token for padding.")
            return
        raise ModelLoadError(
            f"Tokenizer for {self.config.model_id} has no pad, eos, or unk token, so inputs "
            "cannot be batched. Set batch_size=1, or use a model with a padding token."
        )

    def _reject_unsupported(self, hf_config: Any) -> None:
        """Fail loudly on architectures whose hidden states mean something else.

        Multimodal (vision + text) models are *not* rejected: called with text only,
        the wrapper is its language model. They are flagged and warned about instead.
        """
        text_cfg = self._text_config(hf_config)
        is_enc_dec = bool(
            _cfg_get(hf_config, "is_encoder_decoder")
            or _cfg_get(hf_config, "is_vision_encoder_decoder")
            or (text_cfg is not hf_config and _cfg_get(text_cfg, "is_encoder_decoder"))
        )
        if is_enc_dec:
            arch = (_cfg_get(hf_config, "architectures") or ["unknown"])[0]
            raise UnsupportedArchitectureError(
                f"{self.config.model_id} ({arch}) is an encoder-decoder model. Its "
                "`hidden_states` are the *encoder* states, while `decoder_hidden_states` are "
                "separate -- saving one as 'the last layer' would be misleading. Decoder-only "
                "and encoder-only models are supported today; see docs/design_decisions.md."
            )
        modalities = [
            name for name in ("vision", "audio")
            if _cfg_get(hf_config, f"{name}_config") is not None
        ]
        if modalities:
            self._multimodal = True
            log.warning(
                "%s is a multimodal model (%s). This pipeline feeds text only, so hidden "
                "states and log-probabilities come from its language model (%s); the %s "
                "tower weights are loaded but never run. Layer count, hidden size and "
                "context length are read from `text_config`.",
                self.config.model_id,
                ", ".join(f"{m}_config" for m in modalities),
                _cfg_get(text_cfg, "model_type") or "text_config",
                "/".join(modalities),
            )

    @staticmethod
    def _text_config(hf_config: Any) -> Any:
        """The language-model sub-config of a composite config, else the config itself.

        Some remote-code configs store ``text_config`` as a plain dict; ``_cfg_get``
        reads either, so callers never need to care.
        """
        sub = _cfg_get(hf_config, "text_config")
        return sub if sub is not None else hf_config

    def _require_causal_lm(self, hf_config: Any) -> None:
        """The causal head only exists for decoder-only architectures."""
        try:
            from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        except Exception:  # pragma: no cover - very old transformers
            return
        model_type = _cfg_get(hf_config, "model_type")
        text_model_type = _cfg_get(self._text_config(hf_config), "model_type")
        if not any(t in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES for t in (model_type, text_model_type) if t):
            raise UnsupportedArchitectureError(
                f"{self.config.model_id} (model_type={model_type!r}) has no causal language-"
                "model head in transformers, so it cannot score next-token log-probabilities. "
                "Use a decoder-only model (GPT-2, Qwen, Llama, OLMo, ...) with head='causal_lm', "
                "or head='base' for hidden states only."
            )

    def _explain_load_failure(self, exc: Exception) -> ModelLoadError:
        """Translate opaque hub errors into something actionable from a notebook."""
        name = type(exc).__name__
        msg = str(exc)
        gated = name in {"GatedRepoError"} or "gated" in msg.lower() or "401" in msg
        missing = name in {"RepositoryNotFoundError"} or "404" in msg

        if gated:
            return ModelLoadError(
                f"Access to '{self.config.model_id}' was denied (gated or private repo).\n"
                "  1. Open https://huggingface.co/" + self.config.model_id + " and accept the licence.\n"
                "  2. Create a token at https://huggingface.co/settings/tokens (read scope).\n"
                "  3. In Colab: key icon in the left sidebar -> add secret named HF_TOKEN, "
                "enable 'Notebook access'. Outside Colab: export HF_TOKEN=...\n"
                f"Original error: {name}: {msg}"
            )
        if missing:
            return ModelLoadError(
                f"Model '{self.config.model_id}' was not found on the Hugging Face Hub. "
                "Check the exact 'org/name' spelling (it is case-sensitive), and if the repo is "
                f"private, configure an HF_TOKEN as described in the README.\nOriginal error: {name}: {msg}"
            )
        if "trust_remote_code" in msg:
            return ModelLoadError(
                f"'{self.config.model_id}' ships custom modelling code. Re-run with "
                "ModelConfig(trust_remote_code=True) only if you trust that repository -- it "
                f"executes arbitrary Python from the Hub.\nOriginal error: {name}: {msg}"
            )
        return ModelLoadError(f"Failed to load '{self.config.model_id}': {name}: {msg}")

    def _detect_revision(self) -> str | None:
        """Best-effort commit SHA of what was actually loaded."""
        for obj in (self._model, self._tokenizer):
            sha = getattr(getattr(obj, "config", obj), "_commit_hash", None)
            if sha:
                return sha
        return self.config.revision

    # -------------------------------------------------------------- accessors

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("ModelService.load() must be called before use.")

    @property
    def model(self) -> Any:
        self._require_loaded()
        return self._model

    @property
    def tokenizer(self) -> Any:
        self._require_loaded()
        return self._tokenizer

    @property
    def device(self) -> str:
        self._require_loaded()
        return self._device  # type: ignore[return-value]

    @property
    def dtype(self) -> Any:
        self._require_loaded()
        return self._dtype

    @property
    def hf_config(self) -> Any:
        self._require_loaded()
        return self._model.config

    @property
    def text_config(self) -> Any:
        """Where layer count / hidden size live: ``text_config`` on multimodal wrappers."""
        return self._text_config(self.hf_config)

    @property
    def multimodal(self) -> bool:
        """True when the loaded checkpoint also carries a vision/audio tower (unused here)."""
        self._require_loaded()
        return self._multimodal

    @property
    def num_hidden_layers(self) -> int:
        """Number of transformer blocks (excludes the embedding output)."""
        cfg = self.text_config
        for attr in ("num_hidden_layers", "n_layer", "num_layers", "n_layers"):
            value = _cfg_get(cfg, attr)
            if isinstance(value, int) and value > 0:
                return value
        raise ModelLoadError(f"Could not determine layer count for {self.config.model_id}.")

    @property
    def hidden_size(self) -> int:
        cfg = self.text_config
        for attr in ("hidden_size", "n_embd", "d_model", "dim"):
            value = _cfg_get(cfg, attr)
            if isinstance(value, int) and value > 0:
                return value
        raise ModelLoadError(f"Could not determine hidden size for {self.config.model_id}.")

    @property
    def architecture(self) -> str | None:
        archs = getattr(self.hf_config, "architectures", None)
        if archs:
            return archs[0]
        return getattr(self.hf_config, "model_type", None)

    # ----------------------------------------------------------- layer picking

    def resolve_layers(self, spec: Any) -> list[tuple[str, int]]:
        """Turn a layer spec into ``[(label, absolute_index), ...]``.

        Accepts an int (negative allowed), the keywords ``"all"``/``"last"``/
        ``"middle"``, or any sequence mixing the two. Order is preserved and
        duplicates are dropped (first occurrence wins).
        """
        n = self.num_hidden_layers
        total = n + 1  # + embedding output at index 0
        out: list[tuple[str, int]] = []

        for label, index in self._expand(spec, n):
            if index < 0:
                index += total
            if not 0 <= index < total:
                raise ValueError(
                    f"Layer {label!r} resolves to index {index}, outside the valid range "
                    f"0..{total - 1} for {self.config.model_id} "
                    f"({n} transformer blocks + 1 embedding output at index 0)."
                )
            if all(existing != index for _, existing in out):
                out.append((label, index))

        if not out:
            raise ValueError(f"Layer spec {spec!r} selected no layers.")
        return out

    @staticmethod
    def _expand(spec: Any, n: int) -> list[tuple[str, int]]:
        if isinstance(spec, str):
            if spec == "all":
                return [(str(i), i) for i in range(n + 1)]
            if spec == "last":
                return [("-1", -1)]
            if spec == "middle":
                return [("middle", n // 2)]
            raise ValueError(f"Unknown layer keyword {spec!r}; expected 'all', 'last', or 'middle'.")
        if isinstance(spec, bool):
            raise ValueError("layers must not be a bool.")
        if isinstance(spec, int):
            return [(str(spec), spec)]
        if isinstance(spec, Sequence):
            out: list[tuple[str, int]] = []
            for item in spec:
                out.extend(ModelService._expand(item, n))
            return out
        raise TypeError(f"Unsupported layers spec type: {type(spec).__name__}")

    # ------------------------------------------------------------ token limits

    def effective_max_length(self, requested: int | None, cap: int) -> int:
        """Resolve the truncation length actually used.

        Takes the model's own context length unless the caller asked for less,
        then applies `cap` so a 128k-context model cannot be handed a sequence
        that would exhaust GPU memory by accident.

        Both `ExtractionConfig` and `ScoringConfig` carry a `max_length_cap` and
        call this, so the log line names the setting, not a config class.
        """
        model_max = self._model_context_length()
        chosen = model_max if requested is None else min(requested, model_max)
        if requested is not None and requested > model_max:
            log.warning(
                "max_length=%d exceeds the model's context length (%d); using %d.",
                requested, model_max, model_max,
            )
        if chosen > cap:
            log.info("Capping max_length %d -> %d (max_length_cap).", chosen, cap)
            chosen = cap
        return max(1, chosen)

    def _model_context_length(self) -> int:
        candidates = []
        tok_max = getattr(self._tokenizer, "model_max_length", None)
        if isinstance(tok_max, int) and tok_max > 0 and tok_max not in _SENTINEL_LENGTHS:
            candidates.append(tok_max)
        cfg = self.text_config
        for attr in ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length"):
            value = _cfg_get(cfg, attr)
            if isinstance(value, int) and value > 0:
                candidates.append(value)
        return min(candidates) if candidates else _DEFAULT_MAX_LENGTH

    # --------------------------------------------------------------- metadata

    def metadata(self) -> dict[str, Any]:
        """Provenance for the run manifest."""
        self._require_loaded()
        return {
            "model_id": self.config.model_id,
            "revision": self._resolved_revision,
            "head": self.config.head,
            "architecture": self.architecture,
            "model_type": getattr(self.hf_config, "model_type", None),
            "text_model_type": _cfg_get(self.text_config, "model_type"),
            "multimodal": self._multimodal,
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "device": self._device,
            "dtype": str(self._dtype).replace("torch.", ""),
            "context_length": self._model_context_length(),
            "pad_token_substituted": self._pad_token_added,
            "device_info": describe_device(self._device or "cpu"),
            "library_versions": library_versions(),
        }


def _cfg_get(cfg: Any, attr: str) -> Any:
    """Read a config attribute from a transformers config object *or* a plain dict."""
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get(attr)
    return getattr(cfg, attr, None)


def library_versions() -> dict[str, str]:
    """Versions that can change results; stamped into every output file."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for module_name, key in (("torch", "torch"), ("transformers", "transformers"), ("numpy", "numpy")):
        try:
            module = __import__(module_name)
            versions[key] = getattr(module, "__version__", "unknown")
        except Exception:  # pragma: no cover
            versions[key] = "not-installed"
    return versions
