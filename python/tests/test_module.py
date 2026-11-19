"""The Python module against the command it wraps.

The module adds nothing to the detector but a way in from Python, so what is
checked is the way in: that a page handed over as a path, a Pillow image or an
array comes back with the lines the `jamil` command gives for the file, that a
pile comes back in order, and that what comes back is laid out as Kraken's
answer is.

Needs a model and pages, named as the Rust checks name them:

    JAMIL_MODEL=model.safetensors JAMIL_GOLDEN=golden_dir pytest python/tests

and the module where Python finds it (`PYTHONPATH=python`, once it is built).
Skipped without them.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

jamil = pytest.importorskip("jamil")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

COMMAND = Path(__file__).resolve().parents[2] / "target" / "release" / "jamil"


@pytest.fixture(scope="session")
def model():
    path = os.environ.get("JAMIL_MODEL")
    if not path:
        pytest.skip("JAMIL_MODEL names no model")
    return jamil.Model(path)


@pytest.fixture(scope="session")
def pages() -> list[str]:
    """Three pages kept without loss, the smallest first: a PNG is the same pixels whoever opens it."""
    golden = os.environ.get("JAMIL_GOLDEN")
    if not golden:
        pytest.skip("JAMIL_GOLDEN names no pages")
    images = [json.loads(p.read_text())["image"] for p in Path(golden).glob("*.json") if not p.name.endswith(".outlines.json")]
    lossless = sorted((p for p in images if p.endswith(".png")), key=os.path.getsize)
    if len(lossless) < 3:
        pytest.skip("fewer than three PNG pages")
    return lossless[:3]


@pytest.fixture(scope="session")
def read(model, pages):
    """Each page read alone from its file: what everything else is held against."""
    return [model.segment(page) for page in pages]


def found(page) -> tuple[list, list]:
    """A page's lines and regions as bare points."""
    lines = [(line.baseline, line.boundary) for line in page.lines]
    regions = [region.boundary for of_a_kind in page.regions.values() for region in of_a_kind]
    return lines, regions


def test_with_no_model_named_the_network_jamil_carries_reads_the_page_as_the_command_does(pages) -> None:
    if not COMMAND.exists():
        pytest.skip("the command is not built")
    carried = jamil.Model()

    page = carried.segment(pages[0])

    ran = subprocess.run([str(COMMAND), pages[0]], capture_output=True, text=True, check=True)
    said = json.loads(ran.stdout)
    assert carried.height == 1800
    assert [(line.baseline, line.boundary) for line in page.lines] == [(line["baseline"], line["boundary"]) for line in said["lines"]]


def test_a_page_s_lines_are_the_command_s(model, pages, read) -> None:
    if not COMMAND.exists():
        pytest.skip("the command is not built")
    ran = subprocess.run([str(COMMAND), os.environ["JAMIL_MODEL"], pages[0]], capture_output=True, text=True, check=True)
    if ("on the GPU" in ran.stderr) != model.on_gpu:
        pytest.skip("the command and the module are not on the same device")

    said = json.loads(ran.stdout)

    lines, regions = found(read[0])
    assert [(line["baseline"], line["boundary"]) for line in said["lines"]] == lines
    assert said["regions"] == regions
    assert tuple(said["size"]) == read[0].size


def test_a_pillow_image_is_read_as_its_file_is(model, pages, read) -> None:
    with Image.open(pages[0]) as image:
        page = model.segment(image)

    assert found(page) == found(read[0])
    assert page.imagename == pages[0]


def test_an_array_is_read_as_the_image_it_holds(model, pages, read) -> None:
    with Image.open(pages[0]) as image:
        pixels = np.asarray(image.convert("RGB"))

    page = model.segment(pixels)

    assert found(page) == found(read[0])
    assert page.imagename is None


def test_an_array_with_no_colour_is_read_as_a_grey_image(model, pages) -> None:
    with Image.open(pages[0]) as image:
        grey = image.convert("L")

    page = model.segment(np.asarray(grey))

    assert found(page) == found(model.segment(grey))


def test_an_array_that_does_not_lie_in_one_piece_is_read_as_what_it_shows(model, pages) -> None:
    """Mirrored by a negative stride: its memory runs the other way from what it shows."""
    with Image.open(pages[0]) as image:
        pixels = np.asarray(image.convert("RGB"))
    mirrored = pixels[:, ::-1]

    page = model.segment(mirrored)

    assert found(page) == found(model.segment(np.ascontiguousarray(mirrored)))


def test_a_pile_comes_back_in_the_order_given_each_page_as_it_is_read_alone(model, pages, read) -> None:
    with Image.open(pages[1]) as image:
        image.load()
    with Image.open(pages[2]) as other:
        pixels = np.asarray(other.convert("RGB"))

    pile = model.segment_many([pages[0], image, pixels, pages[0]])

    assert [found(page) for page in pile] == [found(read[i]) for i in (0, 1, 2, 0)]
    assert [page.size for page in pile] == [read[i].size for i in (0, 1, 2, 0)]


def test_a_pile_may_be_anything_that_hands_out_pages(model, pages, read) -> None:
    pile = model.segment_many(page for page in pages[:2])

    assert [found(page) for page in pile] == [found(page) for page in read[:2]]


def test_a_page_that_cannot_be_read_is_named_and_does_not_stop_the_rest(model, pages, read, tmp_path) -> None:
    missing = str(tmp_path / "no-such-page.png")

    with pytest.raises(RuntimeError, match="1 of 3 pages could not be read.*no-such-page.png"):
        model.segment_many([pages[0], missing, pages[1]])
    pile = model.segment_many([pages[0], missing, pages[1]], errors="none")

    assert pile[1] is None
    assert [found(pile[0]), found(pile[2])] == [found(read[0]), found(read[1])]


def test_a_page_alone_that_cannot_be_read_raises_with_its_name(model, tmp_path) -> None:
    with pytest.raises(RuntimeError, match="no-such-page.png"):
        model.segment(tmp_path / "no-such-page.png")


def test_what_comes_back_is_laid_out_as_kraken_s_answer(model, pages, read) -> None:
    page = read[0]

    assert (page.type, page.text_direction, page.script_detection, page.line_orders) == ("baselines", "horizontal-rl", False, [])
    assert page.imagename == pages[0]
    assert set(page.regions) == set(model.region_types)
    ids = {region.id for of_a_kind in page.regions.values() for region in of_a_kind}
    for kind, of_a_kind in page.regions.items():
        for region in of_a_kind:
            assert region.tags == {"type": [{"type": kind}]}
            assert region.boundary[0] == region.boundary[-1]
    assert page.lines
    for line in page.lines:
        assert line.type == "baselines" and line.text is None
        assert line.tags == {"type": [{"type": model.baseline_type}]}
        assert line.boundary[0] == line.boundary[-1]
        assert set(line.regions) <= ids
        assert all(isinstance(v, int) for point in line.baseline for v in point)
    assert len({line.id for line in page.lines} | ids) == len(page.lines) + len(ids)


def test_lines_read_from_the_left_are_the_same_lines_in_another_order(model, pages, read) -> None:
    page = model.segment(pages[0], text_direction="horizontal-lr")

    key = json.dumps
    assert sorted(map(key, found(page)[0])) == sorted(map(key, found(read[0])[0]))
    assert page.text_direction == "horizontal-lr"


def test_mended_lines_are_the_command_s_mended_lines(model, pages) -> None:
    if not COMMAND.exists():
        pytest.skip("the command is not built")
    ran = subprocess.run([str(COMMAND), os.environ["JAMIL_MODEL"], pages[0], "--mend"], capture_output=True, text=True, check=True)
    if ("on the GPU" in ran.stderr) != model.on_gpu:
        pytest.skip("the command and the module are not on the same device")

    page = model.segment(pages[0], mend=True)

    said = json.loads(ran.stdout)
    assert [(line["baseline"], line["boundary"]) for line in said["lines"]] == found(page)[0]


def test_mending_joins_lines_and_makes_none(model, pages, read) -> None:
    """Whatever it joins, it is never more lines than the tracing left, and every page is still read."""
    mended = model.segment_many(pages, mend=True)

    assert [len(page.lines) <= len(as_traced.lines) for page, as_traced in zip(mended, read)] == [True] * len(pages)
    assert all(page.lines for page in mended)
    assert found(model.segment(pages[0], mend=True)) == found(mended[0])


def test_the_function_is_the_method_called_as_kraken_s_is(model, pages, read) -> None:
    page = jamil.segment(pages[0], model=model)

    assert found(page) == found(read[0])


def test_kraken_takes_what_to_kraken_makes(read) -> None:
    containers = pytest.importorskip("kraken.containers")

    theirs = read[0].to_kraken()

    assert isinstance(theirs, containers.Segmentation)
    assert all(isinstance(line, containers.BaselineLine) for line in theirs.lines)
    assert [(line.baseline, line.boundary) for line in theirs.lines] == found(read[0])[0]
    assert [line.id for line in theirs.lines] == [line.id for line in read[0].lines]
    assert {kind: [region.id for region in regions] for kind, regions in theirs.regions.items()} == {
        kind: [region.id for region in regions] for kind, regions in read[0].regions.items()
    }


@pytest.mark.parametrize("direction", ["vertical-lr", "vertical-rl", "sideways"])
def test_a_direction_that_is_not_traced_is_refused(model, pages, direction) -> None:
    with pytest.raises(ValueError):
        model.segment(pages[0], text_direction=direction)


def test_what_is_not_a_page_is_refused_by_what_it_is(model) -> None:
    with pytest.raises(TypeError, match="path, a Pillow image, or an array"):
        model.segment(42)
    with pytest.raises(TypeError, match="float32"):
        model.segment(np.zeros((40, 30, 3), dtype="float32"))
    with pytest.raises(ValueError, match="shape"):
        model.segment(np.zeros((40, 30, 4), dtype="uint8"))


def test_one_path_is_not_a_pile_of_its_letters(model, pages) -> None:
    with pytest.raises(TypeError, match="segment_many takes several pages"):
        model.segment_many(pages[0])


def test_a_model_that_is_not_there_says_which() -> None:
    with pytest.raises(RuntimeError, match="no-such-model.safetensors"):
        jamil.Model("no-such-model.safetensors")


def test_a_model_kept_to_the_processor_says_so_and_warns_of_nothing(recwarn) -> None:
    path = os.environ.get("JAMIL_MODEL")
    if not path:
        pytest.skip("JAMIL_MODEL names no model")

    model = jamil.Model(path, device="cpu")

    assert not model.on_gpu
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_a_device_or_arithmetic_nobody_offers_is_refused() -> None:
    with pytest.raises(ValueError, match="device"):
        jamil.Model("model.safetensors", device="tpu")
    with pytest.raises(ValueError, match="arithmetic"):
        jamil.Model("model.safetensors", arithmetic="rough")
