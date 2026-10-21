"""Dependency-light scheduling, aggregation, and text-chunk helpers."""
from __future__ import annotations

import math
import re
from typing import Callable, Iterable


def _percentile25(values: list[float]) -> float:
    xs = sorted(values)
    pos = (len(xs) - 1) * 0.25
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def longav_aggregate(values: Iterable[float]) -> tuple[float, float, float]:
    """Return scaled score, raw score, and fraction of saturated inputs."""
    vals = [float(x) for x in values]
    if not vals or not all(math.isfinite(x) for x in vals):
        raise ValueError("scores must be non-empty and finite")
    raw = 0.8 * (sum(vals) / len(vals)) + 0.2 * _percentile25(vals)
    scaled_items = [min(1.0, max(0.0, 2.5 * max(x, 0.0))) for x in vals]
    score = 0.8 * (sum(scaled_items) / len(scaled_items)) + 0.2 * _percentile25(scaled_items)
    saturation = sum(x >= 1.0 for x in scaled_items) / len(scaled_items)
    return score, raw, saturation


def sliding_windows(frame_count: int, window: int = 8, stride: int = 4) -> list[list[int]]:
    if frame_count < 1 or window < 1 or stride < 1:
        raise ValueError("frame_count, window, and stride must be positive")
    if frame_count <= window:
        return [list(range(frame_count))]
    starts = list(range(0, frame_count - window + 1, stride))
    tail = frame_count - window
    if starts[-1] != tail:
        starts.append(tail)
    return [list(range(start, start + window)) for start in starts]


def uniform_indices(frame_count: int, maximum: int) -> list[int]:
    if frame_count < 1 or maximum < 1:
        raise ValueError("frame_count and maximum must be positive")
    if frame_count <= maximum:
        return list(range(frame_count))
    if maximum == 1:
        return [frame_count // 2]
    return sorted({round(i * (frame_count - 1) / (maximum - 1)) for i in range(maximum)})


def chunk_text(text: str, encode: Callable[[str], list[int]], decode: Callable[[list[int]], str], max_content_tokens: int = 30) -> tuple[list[str], list[dict]]:
    """Split on sentence/clause boundaries, then exact tokenizer boundaries.

    The ViCLIP encoder prepends/appends two special tokens, so the default
    content budget is 30 for its 32-token context. No source text is truncated.
    """
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return [], []
    units = [u.strip() for u in re.split(r"(?<=[.!?;:])\s+", text) if u.strip()]
    chunks: list[str] = []
    fallback: list[dict] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    def split_tokens(unit: str) -> None:
        token_ids = encode(unit)
        if len(token_ids) <= max_content_tokens:
            chunks.append(unit)
            return
        pieces = [decode(token_ids[i:i + max_content_tokens]).strip()
                  for i in range(0, len(token_ids), max_content_tokens)]
        pieces = [p for p in pieces if p]
        split_method = "token_boundaries"
        if not pieces or any(len(encode(p)) > max_content_tokens for p in pieces):
            # Some BPE decoders normalize punctuation/spacing when decoding a
            # token slice. Re-encoding that text can therefore exceed the
            # original token slice budget. Fall back to source-text boundaries
            # rather than dropping the whole sample.
            pieces = []
            remaining = unit
            while remaining:
                lo, hi, best = 1, len(remaining), 0
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if len(encode(remaining[:mid])) <= max_content_tokens:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if best == 0:
                    raise ValueError("unable to split ViCLIP text below its token budget")
                end = best
                if end < len(remaining):
                    boundary = remaining.rfind(" ", 0, end + 1)
                    if boundary >= max(1, end // 2):
                        end = boundary
                piece = remaining[:end].strip()
                if piece:
                    if len(encode(piece)) > max_content_tokens:
                        piece = remaining[:best].strip()
                    pieces.append(piece)
                remaining = remaining[end:].lstrip()
            split_method = "source_text_boundaries"
        chunks.extend(pieces)
        fallback.append({"source_unit": unit, "token_boundary_parts": len(pieces),
                         "split_method": split_method})

    for unit in units:
        unit_n = len(encode(unit))
        if unit_n > max_content_tokens:
            flush()
            split_tokens(unit)
            continue
        candidate = f"{current} {unit}".strip()
        if current and len(encode(candidate)) > max_content_tokens:
            flush()
        current = f"{current} {unit}".strip()
    flush()
    if any(len(encode(piece)) > max_content_tokens for piece in chunks):
        raise ValueError("a ViCLIP text chunk exceeds the available token budget")
    return chunks, fallback
