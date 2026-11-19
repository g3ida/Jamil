"""Jamil: a model in, a page's lines of text out"""

from os import PathLike
from typing import Any, Iterable, Literal, Optional, Union

__version__: str

# A path, a Pillow image, or an array of uint8: (rows, columns, 3) or (rows, columns).
Page = Union[str, PathLike[str], Any]
TextDirection = Literal["horizontal-rl", "horizontal-lr"]
Points = list[list[int]]

class BaselineLine:
    id: str
    type: Literal["baselines"]
    baseline: Points
    boundary: Points
    tags: dict[str, list[dict[str, str]]]
    regions: list[str]
    imagename: Optional[Any]
    text: None
    base_dir: None
    split: None
    language: None

class Region:
    id: str
    boundary: Points
    tags: dict[str, list[dict[str, str]]]
    imagename: Optional[Any]
    language: None

class Segmentation:
    type: Literal["baselines"]
    imagename: Optional[Any]
    text_direction: TextDirection
    script_detection: Literal[False]
    lines: list[BaselineLine]
    regions: dict[str, list[Region]]
    line_orders: list[list[int]]
    language: None
    size: tuple[int, int]
    seconds: dict[str, float]
    def to_kraken(self) -> Any: ...

class Model:
    on_gpu: bool
    height: int
    baseline_type: str
    region_types: list[str]
    def __init__(
        self,
        path: Optional[Union[str, PathLike[str]]] = None,
        *,
        device: Literal["auto", "cuda", "cpu"] = "auto",
        arithmetic: Literal["exact", "tf32", "half"] = "exact",
    ) -> None: ...
    def segment(self, image: Page, *, text_direction: TextDirection = "horizontal-rl", mend: bool = False) -> Segmentation: ...
    def segment_many(
        self,
        images: Iterable[Page],
        *,
        text_direction: TextDirection = "horizontal-rl",
        mend: bool = False,
        errors: Literal["raise", "none"] = "raise",
    ) -> list[Optional[Segmentation]]: ...

def segment(image: Page, *, model: Model, text_direction: TextDirection = "horizontal-rl", mend: bool = False) -> Segmentation: ...
