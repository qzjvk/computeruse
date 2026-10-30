"""LocateAnything-3B worker, tuned for legacy NVIDIA GPUs (e.g. GTX 1070).

Key compatibility decisions for Pascal (sm_61):
  * weights run in float16  (sm_61 has no bfloat16 support)
  * attention uses the SDPA path -> PyTorch's "math" backend (flash-attn /
    mem-efficient kernels and MagiAttention require sm_80+/sm_75+ respectively)
  * cuDNN is disabled for sm < 75 so convs (MoonViT patch embed) fall back to
    PyTorch's native CUDA kernels, which are built for sm_61
  * the image processor's ``in_token_limit`` is capped (default 2048) to keep
    the prefill / logits footprint small on 8 GB VRAM
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Union

# Reduce CUDA allocator fragmentation (must be set before first CUDA call).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

ImageLike = Union[Image.Image, str]


def _patch_sdpa_to_math() -> None:
    """Force SDPA to use the math backend for the whole process.

    Needed on legacy GPUs (sm < 75): the flash/mem-efficient kernels don't
    exist there, and PyTorch <= 2.5 can raise instead of silently falling
    back. The math backend is arch-independent (just slower).
    """
    fn = torch.nn.functional
    if getattr(fn, "_cu_math_sdpa_patched", False):
        return
    orig = fn.scaled_dot_product_attention
    try:
        ctx_factory = torch.nn.attention.sdpa_kernel
    except AttributeError:  # torch < 2.3
        ctx_factory = None

    def patched(query, key, value, *args, **kwargs):
        if ctx_factory is not None:
            with ctx_factory(
                torch.nn.attention.SDPBackend.MATH,
            ):
                return orig(query, key, value, *args, **kwargs)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            return orig(query, key, value, *args, **kwargs)

    fn.scaled_dot_product_attention = patched  # type: ignore[attr-defined]
    fn._cu_math_sdpa_patched = True  # type: ignore[attr-defined]


def pick_device(preferred: Optional[str] = None) -> torch.device:
    """Return ``preferred`` if given, else the GPU with the most free memory."""
    if preferred is not None:
        return torch.device(preferred if preferred.startswith("cuda") else f"cuda:{preferred}")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best_idx, best_free = None, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_idx, best_free = i, free
    return torch.device(f"cuda:{best_idx}")


def gpu_capability(device: torch.device) -> Optional[tuple[int, int]]:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        return torch.cuda.get_device_capability(idx)
    return None


@dataclass
class GroundingResult:
    """Parsed grounding output."""

    answer: str
    points: list[tuple[float, float]] = field(default_factory=list)   # absolute image px
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)  # x1,y1,x2,y2 px
    image_size: tuple[int, int] = (0, 0)  # (w, h) of the image that was grounded
    seconds: float = 0.0
    stats: str = ""

    @property
    def center(self) -> Optional[tuple[float, float]]:
        """Best click target: first point, else center of first box."""
        if self.points:
            return self.points[0]
        if self.boxes:
            x1, y1, x2, y2 = self.boxes[0]
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return None


class LocateAnythingWorker:
    """Stateful worker that loads the model once and serves grounding queries."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        attn: Optional[str] = None,
        in_token_limit: Optional[int] = None,  # None: auto (1024 on sm<75, 2048 elsewhere)
        vit_cpu: bool | str = "auto",  # "auto": True on sm < 8 (8 GB cards)
        verbose: bool = True,
    ):
        self.device = pick_device(device)
        cap = gpu_capability(self.device)
        if in_token_limit is None:
            # Legacy 8 GB GPUs: the prefill logits copy dominates the VRAM
            # budget, so keep vision tokens low (see README, GTX 1070 notes).
            in_token_limit = 1024 if (cap is not None and cap[0] < 7) else 2048

        # Pascal (sm_6x) has no bf16 support -> fp16 there.
        if dtype is None:
            dtype = torch.bfloat16 if (cap is not None and cap[0] >= 8) else torch.float16
        self.dtype = dtype
        self.cap = cap

        # Default attention: SDPA works everywhere (math backend on Pascal).
        # flash_attention_2 / magi are only offered on sm_80+ and only if the
        # packages are installed; we never select them implicitly.
        self.attn = attn or "sdpa"
        if self.attn in ("flash_attention_2", "magi") and (cap is None or cap[0] < 8):
            raise ValueError(f"attn={self.attn!r} requires sm_80+ (got {cap})")
        self.verbose = verbose

        # On sm < 75 the bundled cuDNN (9.x) has no kernels; force native convs.
        if cap is not None and cap[0] < 7:
            torch.backends.cudnn.enabled = False
            # PyTorch <= 2.5's SDPA dispatcher can raise ("FlashAttention only
            # supports Ampere GPUs or newer") on Pascal instead of falling back.
            # Force the math backend for this whole process.
            _patch_sdpa_to_math()

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config._attn_implementation = self.attn
        config.text_config._attn_implementation = self.attn
        # MoonViT (vision encoder): sdpa is the portable choice.
        if getattr(config, "vision_config", None) is not None:
            config.vision_config._attn_implementation = self.attn

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        # Cap vision tokens to fit the target VRAM (see module docstring).
        try:
            self.processor.image_processor.in_token_limit = in_token_limit
        except AttributeError:
            pass
        self.in_token_limit = in_token_limit

        # On 8 GB cards the full fp16 model (≈7.2 GB) plus CUDA context and
        # activations just barely overflows. The MoonViT vision encoder
        # (≈0.4 B) only runs once per query, so keep it on the CPU and move
        # only the LLM + MLP to the GPU (never copy the ViT to the GPU).
        if vit_cpu == "auto":
            vit_cpu = cap is not None and cap[0] < 8
        vit_cpu = bool(vit_cpu) and self.device.type == "cuda"

        model = AutoModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if vit_cpu:
            model.vision_model = model.vision_model.to("cpu", torch.float32)
            for name, child in list(model.named_children()):
                if name != "vision_model":
                    child.to(self.device)
        else:
            model = model.to(self.device)
        self.model = model.eval()
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

        if vit_cpu:
            _orig_extract = self.model.extract_feature
            _llm_embed = self.model.language_model.get_input_embeddings()

            def _extract_feature_offloaded(pixel_values, image_grid_hws):
                pdev = next(self.model.vision_model.parameters()).device
                pdtype = next(self.model.vision_model.parameters()).dtype
                if isinstance(image_grid_hws, torch.Tensor):
                    image_grid_hws = image_grid_hws.to(pdev)
                out = _orig_extract(pixel_values.to(pdev, pdtype), image_grid_hws)
                dev, dt = _llm_embed.weight.device, _llm_embed.weight.dtype
                # MoonViT returns one tensor per image (list).
                if isinstance(out, (list, tuple)):
                    return [t.to(dev, dt) for t in out]
                return out.to(dev, dt)

            self.model.extract_feature = _extract_feature_offloaded
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            if self.verbose:
                print("[worker] MoonViT vision encoder offloaded to CPU (fp32)")

        if self.verbose:
            idx = self.device.index if self.device.type == "cuda" else -1
            dev_name = torch.cuda.get_device_name(idx) if self.device.type == "cuda" else "cpu"
            free_gb, _ = (
                torch.cuda.mem_get_info(self.device.index)
                if self.device.type == "cuda"
                else (0, 0)
            )
            print(
                f"[worker] device=cuda:{idx} ({dev_name}, sm_{cap[0]}{cap[1]}) "
                f"dtype={dtype} attn={self.attn} in_token_limit={in_token_limit} "
                f"free_vram={free_gb / 1e9:.2f}G"
            )

    # ------------------------------------------------------------------ core

    def _prepare_inputs(self, image: Image.Image, question: str):
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        )
        return inputs

    @torch.no_grad()
    def predict(
        self,
        image: ImageLike,
        question: str,
        generation_mode: str = "hybrid",   # "fast" (MTP) | "slow" (AR) | "hybrid"
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        verbose: Optional[bool] = None,
    ) -> dict:
        """Run one grounding query. Returns {"answer": str, "history": ..., "stats": ...}."""
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        verbose = self.verbose if verbose is None else verbose

        def _run():
            # Inputs are prepared under the processor's current limit: on an
            # OOM retry the limit is halved first, so the image is actually
            # re-shrunk, not just re-generated.
            inputs = self._prepare_inputs(image, question)
            pixel_values = inputs["pixel_values"].to(self.device, self.dtype)
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            image_grid_hws = inputs.get("image_grid_hws", None)
            return self.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_hws=image_grid_hws,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else None,
                repetition_penalty=1.1,
                verbose=verbose,
            )

        t0 = time.time()
        # The budget assumed at load time may no longer fit (the card is
        # shared, or the allocator is fragmented after many passes). Work
        # down a ladder of budgets (4096 -> 2048 -> 1024); each rung is
        # half the prefill transient of the one before.
        old = self.in_token_limit
        nominal = old or 2048
        ladder = [nominal]
        while ladder[-1] > 1024:
            ladder.append(ladder[-1] // 2)
        last_err = None
        response = None
        try:
            for i, lim in enumerate(ladder):
                if lim != old:
                    self.in_token_limit = lim
                    try:
                        self.processor.image_processor.in_token_limit = lim
                    except AttributeError:
                        pass
                try:
                    response = _run()
                    break
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    torch.cuda.empty_cache()
                    last_err = e
                    if i + 1 >= len(ladder):
                        break
                    print(f"[worker] prefill OOM at in_token_limit={lim}; "
                          f"retrying at {ladder[i + 1]} (card shared or "
                          f"fragmented)", flush=True)
        finally:
            if self.in_token_limit != old:
                self.in_token_limit = old
                try:
                    self.processor.image_processor.in_token_limit = old
                except AttributeError:
                    pass
        if response is None:
            raise last_err
        # Reclaim the prefill's transient memory before the next query.
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        dt = time.time() - t0

        result: dict = {"answer": response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        result["seconds"] = dt
        return result

    # ------------------------------------------------------- convenience API

    def point(self, image: ImageLike, phrase: str, **kwargs) -> GroundingResult:
        """Point-based GUI grounding (preferred for click targets)."""
        return self.ground(image, phrase, output_type="point", **kwargs)

    def ground(self, image: ImageLike, phrase: str, output_type: str = "point", **kwargs) -> GroundingResult:
        """GUI grounding: 'point' -> <box><x><y></box>, 'box' -> 4-coord box."""
        if output_type == "point":
            prompt = f"Point to: {phrase}."
        else:
            prompt = f"Locate the region that matches the following description: {phrase}."
        res = self.predict(image, prompt, **kwargs)
        img = image if isinstance(image, Image.Image) else Image.open(image)
        w, h = img.size
        from .grounding import parse_points, parse_boxes  # local import to avoid cycle
        return GroundingResult(
            answer=res["answer"],
            points=parse_points(res["answer"], w, h),
            boxes=parse_boxes(res["answer"], w, h),
            image_size=(w, h),
            seconds=res.get("seconds", 0.0),
            stats=res.get("stats", ""),
        )

    def detect(self, image: ImageLike, categories: list[str], **kwargs) -> GroundingResult:
        """Open-set detection / document layout: 'Locate all the instances...'."""
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        res = self.predict(image, prompt, **kwargs)
        img = image if isinstance(image, Image.Image) else Image.open(image)
        w, h = img.size
        from .grounding import parse_boxes
        return GroundingResult(
            answer=res["answer"],
            boxes=parse_boxes(res["answer"], w, h),
            image_size=(w, h),
            seconds=res.get("seconds", 0.0),
            stats=res.get("stats", ""),
        )