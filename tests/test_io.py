import torch

from zeroth_grad.cli import main
from zeroth_grad.data import concept_bank
from zeroth_grad.io import load_checkpoint, save_checkpoint
from zeroth_grad.mlp import shared_init


def test_roundtrip_safetensors_and_pt(tmp_path):
    weights = shared_init(seed=4, h=8, d_out=8)
    meta = {"skills": "1,3", "split": "train", "kind": "single", "base_h1": "8", "base_h2": "8"}
    for suffix in (".safetensors", ".pt"):
        path = tmp_path / f"model{suffix}"
        save_checkpoint(path, weights, meta)
        loaded, loaded_meta = load_checkpoint(path)
        for key in weights:
            assert torch.allclose(loaded[key], weights[key])
            assert not loaded[key].requires_grad
        assert loaded_meta["skills"] == "1,3"
        assert loaded_meta["base_h1"] == "8"


def test_cli_help():
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0


def test_cli_widen_and_compile_roundtrip_shapes(tmp_path):
    concepts = concept_bank(0)
    weights = shared_init(seed=5, h=8, d_out=8)
    src = tmp_path / "model.safetensors"
    wide = tmp_path / "wide.safetensors"
    save_checkpoint(
        src,
        weights,
        {"skills": "0", "base_h1": "8", "base_h2": "8", "h1": "8", "h2": "8", "split": "train", "kind": "single"},
    )
    main(["widen", "--ckpt", str(src), "--out", str(wide), "--copies", "4", "--neurons", "1", "--layer", "h1", "--shuffle"])
    loaded, meta = load_checkpoint(wide)
    assert loaded["w1"].shape[0] == 11
    assert meta["base_h1"] == "8"
    # A randomly initialized compiler still collapses exact duplicates.
    from zeroth_grad.compiler import WeightCompiler
    from zeroth_grad.train import save_compiler

    compiler_path = tmp_path / "compiler.pt"
    save_compiler(compiler_path, WeightCompiler())
    out = tmp_path / "slim.safetensors"
    main(["compile", "compress", "--compiler", str(compiler_path), "--ckpt", str(wide), "--out", str(out)])
    slim, slim_meta = load_checkpoint(out)
    assert slim["w1"].shape[0] == 8
    assert slim_meta["edit"] == "compress"
    examples = tmp_path / "examples.pt"
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(500, 16, generator=generator)
    y = (x @ concepts[3] > 0).float()
    torch.save({"x": x, "y": y}, examples)
    injected = tmp_path / "injected.safetensors"
    main(
        [
            "compile",
            "inject",
            "--compiler",
            str(compiler_path),
            "--ckpt",
            str(src),
            "--examples",
            str(examples),
            "--head",
            "3",
            "--out",
            str(injected),
        ]
    )
    _, injected_meta = load_checkpoint(injected)
    assert injected_meta["edit"] == "inject"
    assert injected_meta["head"] == "3"

    others = []
    for index, skill in enumerate((1, 2)):
        path = tmp_path / f"other{index}.pt"
        save_checkpoint(path, shared_init(seed=6 + index, h=8, d_out=8), {"skills": str(skill)})
        others.append(str(path))
    merged = tmp_path / "merged.safetensors"
    main(["compile", "merge", "--compiler", str(compiler_path), "--ckpts", str(src), *others, "--out", str(merged)])
    merged_weights, merged_meta = load_checkpoint(merged)
    assert merged_weights["w1"].shape == weights["w1"].shape
    assert merged_meta["edit"] == "merge"
    assert merged_meta["skills"] == "0,1,2"
    assert merged_meta["parents"] == "3"
    pair = tmp_path / "pair.safetensors"
    main(["compile", "merge", "--compiler", str(compiler_path), "--ckpt-a", str(src), "--ckpt-b", others[0], "--out", str(pair)])
    assert load_checkpoint(pair)[1]["skills"] == "0,1"
