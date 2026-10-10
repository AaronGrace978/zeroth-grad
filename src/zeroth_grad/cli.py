"""Command line for the weight-space compiler."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from zeroth_grad.data import estimate_direction, format_skills, parse_skills
from zeroth_grad.edit import expand_h1, expand_h2
from zeroth_grad.io import load_checkpoint, save_checkpoint
from zeroth_grad.train import load_compiler


def _load_phi(args: argparse.Namespace, d_in: int) -> tuple[torch.Tensor, int]:
    if args.examples is not None:
        blob = torch.load(args.examples, map_location="cpu", weights_only=False)
        phi = estimate_direction(blob["x"], blob["y"])
    elif args.phi is not None:
        phi = torch.load(args.phi, map_location="cpu", weights_only=False)
        phi = phi if isinstance(phi, torch.Tensor) else phi["phi"]
    elif args.concept_index is not None:
        if args.concepts is None:
            raise SystemExit("inject --concept-index requires --concepts")
        concepts = torch.load(args.concepts, map_location="cpu", weights_only=False)["concepts"]
        phi = concepts[int(args.concept_index)]
    else:
        raise SystemExit("inject needs --examples, --phi, or --concept-index")
    if phi.numel() != d_in:
        raise SystemExit(f"feature direction has {phi.numel()} dims, checkpoint expects {d_in}")
    if args.head is None:
        raise SystemExit("inject requires --head")
    return phi.view(-1).float(), int(args.head)


def _cmd_populate(args: argparse.Namespace) -> None:
    from zeroth_grad.zoo import populate

    populate(args.out, steps=args.steps)


def _cmd_train(args: argparse.Namespace) -> None:
    from zeroth_grad.train import train_from_dir

    train_from_dir(args.checkpoints, args.out, steps=args.steps, seed=args.seed)


def _cmd_eval(args: argparse.Namespace) -> None:
    from zeroth_grad.evaluate import evaluate_all

    compiler = load_compiler(args.compiler)
    evaluate_all(compiler, args.checkpoints, out=args.out)
    print(Path(args.out).read_text())


def _cmd_widen(args: argparse.Namespace) -> None:
    weights, meta = load_checkpoint(args.ckpt)
    base_h1 = meta.get("base_h1", str(weights["w1"].shape[0]))
    base_h2 = meta.get("base_h2", str(weights["w2"].shape[0]))
    copies = int(args.copies)
    if copies < 2:
        raise SystemExit("--copies must be at least 2")
    extras = copies - 1
    generator = torch.Generator().manual_seed(args.seed)
    if args.layer in ("h1", "both"):
        top = torch.argsort(weights["w2"].norm(dim=0), descending=True)[: args.neurons].tolist()
        sources = [index for index in top for _ in range(extras)]
        weights = expand_h1(weights, sources)
    if args.layer in ("h2", "both"):
        top = torch.argsort(weights["w3"].norm(dim=0), descending=True)[: args.neurons].tolist()
        sources = [index for index in top for _ in range(extras)]
        weights = expand_h2(weights, sources)
    if args.shuffle:
        from zeroth_grad.align import permute_h1, permute_h2

        perm1 = torch.randperm(weights["w1"].shape[0], generator=generator)
        perm2 = torch.randperm(weights["w2"].shape[0], generator=generator)
        weights = permute_h2(permute_h1(weights, perm1), perm2)
    meta["h1"] = str(weights["w1"].shape[0])
    meta["h2"] = str(weights["w2"].shape[0])
    meta["base_h1"] = base_h1
    meta["base_h2"] = base_h2
    save_checkpoint(args.out, weights, meta)
    print(f"wrote {args.out} h1={meta['h1']} h2={meta['h2']} base_h1={meta.get('base_h1')}")


def _compile(args: argparse.Namespace) -> None:
    compiler = load_compiler(args.compiler)
    compiler.eval()
    with torch.no_grad():
        if args.op == "inject":
            weights, meta = load_checkpoint(args.ckpt)
            phi, head = _load_phi(args, int(weights["w1"].shape[1]))
            edited = compiler.inject(weights, phi, head)
            meta = dict(meta)
            meta["edit"] = "inject"
            meta["head"] = str(head)
            save_checkpoint(args.out, edited, meta)
        elif args.op == "compress":
            weights, meta = load_checkpoint(args.ckpt)
            if args.keep_h1 is not None:
                k1 = args.keep_h1
            elif "base_h1" in meta:
                k1 = int(meta["base_h1"])
            else:
                raise SystemExit("compress needs --keep-h1 or base_h1 metadata")
            if args.keep_h2 is not None:
                k2 = args.keep_h2
            elif "base_h2" in meta:
                k2 = int(meta["base_h2"])
            else:
                k2 = int(weights["w2"].shape[0])
            edited = compiler.compress(weights, k1, k2)
            meta = dict(meta)
            meta["edit"] = "compress"
            meta["h1"] = str(edited["w1"].shape[0])
            meta["h2"] = str(edited["w2"].shape[0])
            save_checkpoint(args.out, edited, meta)
        elif args.op == "merge":
            left, meta_a = load_checkpoint(args.ckpt_a)
            right, meta_b = load_checkpoint(args.ckpt_b)
            edited = compiler.merge(left, right)
            skills = sorted(set(parse_skills(meta_a.get("skills", "")) + parse_skills(meta_b.get("skills", ""))))
            meta = dict(meta_a)
            meta["edit"] = "merge"
            meta["skills"] = format_skills(skills)
            meta["kind"] = "merged"
            save_checkpoint(args.out, edited, meta)
        else:
            raise SystemExit(f"unknown compile op {args.op}")
    print(f"wrote {args.out}")


def main(argv: list[str] | None = None) -> None:
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(prog="zeroth-grad", description="Edit neural checkpoints in weight space.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    populate = sub.add_parser("populate", help="Train the model zoo")
    populate.add_argument("--out", type=Path, default=Path("checkpoints"))
    populate.add_argument("--steps", type=int, default=300)
    populate.set_defaults(func=_cmd_populate)

    train = sub.add_parser("train", help="Fit the compiler on the zoo")
    train.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    train.add_argument("--out", type=Path, default=Path("runs/compiler.pt"))
    train.add_argument("--steps", type=int, default=600)
    train.add_argument("--seed", type=int, default=0)
    train.set_defaults(func=_cmd_train)

    evaluate = sub.add_parser("eval", help="Compare the compiler to baselines")
    evaluate.add_argument("--compiler", type=Path, required=True)
    evaluate.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    evaluate.add_argument("--out", type=Path, default=Path("runs/metrics.json"))
    evaluate.set_defaults(func=_cmd_eval)

    widen = sub.add_parser("widen", help="Duplicate neurons to build a compress input")
    widen.add_argument("--ckpt", type=Path, required=True)
    widen.add_argument("--out", type=Path, required=True)
    widen.add_argument("--copies", type=int, default=4)
    widen.add_argument("--neurons", type=int, default=2)
    widen.add_argument("--layer", choices=("h1", "h2", "both"), default="both")
    widen.add_argument("--shuffle", action="store_true")
    widen.add_argument("--seed", type=int, default=0)
    widen.set_defaults(func=_cmd_widen)

    compile_parser = sub.add_parser("compile", help="Apply a trained compiler to a checkpoint")
    compile_parser.add_argument("op", choices=("inject", "compress", "merge"))
    compile_parser.add_argument("--compiler", type=Path, default=Path("runs/compiler.pt"))
    compile_parser.add_argument("--ckpt", type=Path)
    compile_parser.add_argument("--ckpt-a", type=Path)
    compile_parser.add_argument("--ckpt-b", type=Path)
    compile_parser.add_argument("--out", type=Path, required=True)
    compile_parser.add_argument("--examples", type=Path)
    compile_parser.add_argument("--phi", type=Path)
    compile_parser.add_argument("--concepts", type=Path)
    compile_parser.add_argument("--concept-index", type=int)
    compile_parser.add_argument("--head", type=int)
    compile_parser.add_argument("--keep-h1", type=int)
    compile_parser.add_argument("--keep-h2", type=int)
    compile_parser.set_defaults(func=_compile)

    args = parser.parse_args(argv)
    args.func(args)
