"""Command line for the weight-space compiler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from zeroth_grad.compiler import WeightCompiler
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


def _merge_paths(args: argparse.Namespace) -> list[Path]:
    if args.ckpts:
        if args.ckpt_a is not None or args.ckpt_b:
            raise SystemExit("merge takes either --ckpts or --ckpt-a/--ckpt-b, not both")
        paths = list(args.ckpts)
    else:
        if args.ckpt_a is None or not args.ckpt_b:
            raise SystemExit("merge needs --ckpt-a and --ckpt-b, or --ckpts")
        paths = [args.ckpt_a, *args.ckpt_b]
    if len(paths) < 2:
        raise SystemExit("merge needs at least two checkpoints")
    return paths


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


def _cmd_digits_populate(args: argparse.Namespace) -> None:
    from zeroth_grad.digits import populate_digits

    populate_digits(
        args.out,
        hidden=args.hidden,
        steps=args.steps,
        sparsity=args.sparsity,
        seeds=tuple(range(args.seeds)),
        data_seed=args.data_seed,
    )


def _cmd_digits_eval(args: argparse.Namespace) -> None:
    from zeroth_grad.digits import evaluate_digits

    compiler = load_compiler(args.compiler) if args.compiler is not None else None
    metrics = evaluate_digits(args.checkpoints, compiler=compiler, out=args.out, subsets_per_size=args.subsets)
    summary = {key: metrics[key] for key in ("width", "seeds", "ten_way", "merge_k")}
    print(json.dumps(summary, indent=2))


def _cmd_extensions(args: argparse.Namespace) -> None:
    from zeroth_grad.extensions import run_extensions

    run_extensions(
        checkpoints=args.checkpoints,
        compiler=args.compiler,
        digits=args.digits,
        digits_w32=args.digits_w32,
        digits_dense=args.digits_dense,
        widths=tuple(args.widths),
        out=args.out,
        sections=tuple(args.sections),
    )
    print(f"wrote {args.out}")


def _load_examples(path: Path) -> tuple[torch.Tensor, torch.Tensor | None]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, torch.Tensor):
        return blob.float(), None
    y = blob.get("y")
    return blob["x"].float(), (y.float() if y is not None else None)


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


def _prior_compiler(args: argparse.Namespace) -> WeightCompiler:
    """Untrained compiler sized to the checkpoint: the geometric prior with no learned residual."""
    if args.op != "merge":
        raise SystemExit("--prior applies to compile merge only")
    weights, _ = load_checkpoint(_merge_paths(args)[0])
    return WeightCompiler(d_in=int(weights["w1"].shape[1]), d_out=int(weights["w3"].shape[0]))


def _compile(args: argparse.Namespace) -> None:
    if args.op == "fold" or (args.op == "inject" and args.rank is not None):
        compiler = None
    else:
        compiler = _prior_compiler(args) if args.prior else load_compiler(args.compiler)
        compiler.eval()
    with torch.no_grad():
        if args.op == "inject" and args.rank is not None:
            from zeroth_grad.lowrank import inject_rank_k

            weights, meta = load_checkpoint(args.ckpt)
            if args.examples is None or args.head is None:
                raise SystemExit("rank-k inject needs --examples (with x and y) and --head")
            x, y = _load_examples(args.examples)
            if y is None:
                raise SystemExit("rank-k inject needs labels: --examples must hold {'x', 'y'}")
            edited = inject_rank_k(weights, x, y, int(args.head), rank=int(args.rank), units=args.units)
            meta = dict(meta)
            meta["edit"] = "inject"
            meta["head"] = str(args.head)
            meta["rank"] = str(args.rank)
            meta["h1"] = str(edited["w1"].shape[0])
            meta["h2"] = str(edited["w2"].shape[0])
            skills = sorted(set(parse_skills(meta.get("skills", ""))) | {int(args.head)})
            meta["skills"] = format_skills(skills)
            save_checkpoint(args.out, edited, meta)
        elif args.op == "fold":
            from zeroth_grad.fold import fold_report

            weights, meta = load_checkpoint(args.ckpt)
            if args.examples is None:
                raise SystemExit("fold needs --examples: a calibration batch, a tensor or {'x': ...}")
            x, _ = _load_examples(args.examples)
            heads = parse_skills(args.heads) if args.heads else parse_skills(meta.get("skills", ""))
            if not heads:
                raise SystemExit("fold needs --heads or skills metadata")
            report = fold_report(weights, x, heads, agreement=args.agreement)
            meta = dict(meta)
            meta["edit"] = "fold"
            save_checkpoint(args.out, report["weights"], meta)
            print(
                f"kept {report['kept_h1']}/{report['live_h1_before']} hidden-1 and "
                f"{report['kept_h2']}/{report['live_h2_before']} hidden-2 units, agreement {report['agreement']:.4f}"
            )
        elif args.op == "inject":
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
            paths = _merge_paths(args)
            loaded = [load_checkpoint(path) for path in paths]
            heads = [parse_skills(meta.get("skills", "")) for _, meta in loaded] if args.declare_heads else None
            edited, infos = compiler.merge_many(
                [weights for weights, _ in loaded],
                solver=args.solver,
                expand=args.expand,
                share=not args.no_share,
                heads=heads,
                return_info=True,
            )
            skills = sorted({skill for _, meta in loaded for skill in parse_skills(meta.get("skills", ""))})
            meta = dict(loaded[0][1])
            meta["edit"] = "merge"
            meta["skills"] = format_skills(skills)
            meta["kind"] = "merged"
            meta["parents"] = str(len(paths))
            for stale in ("live_h1", "live_h2"):
                meta.pop(stale, None)
            meta["h1"] = str(edited["w1"].shape[0])
            meta["h2"] = str(edited["w2"].shape[0])
            save_checkpoint(args.out, edited, meta)
            added = sum(info["added_h1"] for info in infos), sum(info["added_h2"] for info in infos)
            if any(added):
                print(f"widened by {added[0]} hidden-1 and {added[1]} hidden-2 units to avoid collisions")
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

    digits_populate = sub.add_parser("digits-populate", help="Train sparse one-vs-rest detectors on 8x8 digits")
    digits_populate.add_argument("--out", type=Path, default=Path("digits"))
    digits_populate.add_argument("--hidden", type=int, default=64)
    digits_populate.add_argument("--steps", type=int, default=400)
    digits_populate.add_argument("--sparsity", type=float, default=3e-3)
    digits_populate.add_argument("--seeds", type=int, default=3, help="number of detector seeds")
    digits_populate.add_argument("--data-seed", type=int, default=0)
    digits_populate.set_defaults(func=_cmd_digits_populate)

    digits_eval = sub.add_parser("digits-eval", help="Merge the digit detectors into 10-way classifiers")
    digits_eval.add_argument("--checkpoints", type=Path, default=Path("digits"))
    digits_eval.add_argument("--compiler", type=Path, help="defaults to the untrained geometric prior")
    digits_eval.add_argument("--out", type=Path, default=Path("runs/digits_metrics.json"))
    digits_eval.add_argument("--subsets", type=int, default=20, help="digit subsets sampled per merge size")
    digits_eval.set_defaults(func=_cmd_digits_eval)

    extensions = sub.add_parser("extensions", help="Benchmark Sinkhorn alignment, capacity, rank-k inject, and transformers")
    extensions.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    extensions.add_argument("--compiler", type=Path, default=Path("runs/compiler.pt"))
    extensions.add_argument("--digits", type=Path, default=Path("digits"))
    extensions.add_argument("--digits-w32", type=Path, default=Path("digits_w32"))
    extensions.add_argument("--digits-dense", type=Path, default=Path("digits_dense"))
    extensions.add_argument("--widths", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096])
    extensions.add_argument(
        "--sections",
        nargs="+",
        choices=("alignment", "capacity", "inject", "transformer"),
        default=["alignment", "capacity", "inject", "transformer"],
    )
    extensions.add_argument("--out", type=Path, default=Path("runs/extensions_metrics.json"))
    extensions.set_defaults(func=_cmd_extensions)

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
    compile_parser.add_argument("op", choices=("inject", "compress", "merge", "fold"))
    compile_parser.add_argument("--compiler", type=Path, default=Path("runs/compiler.pt"))
    compile_parser.add_argument(
        "--prior",
        action="store_true",
        help="merge with an untrained compiler sized to the checkpoints instead of loading --compiler",
    )
    compile_parser.add_argument("--ckpt", type=Path)
    compile_parser.add_argument("--ckpt-a", type=Path)
    compile_parser.add_argument("--ckpt-b", type=Path, nargs="+", help="one or more checkpoints merged into --ckpt-a, in order")
    compile_parser.add_argument("--ckpts", type=Path, nargs="+", help="checkpoints to merge, folded left to right")
    compile_parser.add_argument("--out", type=Path, required=True)
    compile_parser.add_argument("--examples", type=Path)
    compile_parser.add_argument("--phi", type=Path)
    compile_parser.add_argument("--concepts", type=Path)
    compile_parser.add_argument("--concept-index", type=int)
    compile_parser.add_argument("--head", type=int)
    compile_parser.add_argument("--keep-h1", type=int)
    compile_parser.add_argument("--keep-h2", type=int)
    compile_parser.add_argument("--solver", choices=("hungarian", "sinkhorn", "auto"), default="hungarian")
    compile_parser.add_argument("--expand", action="store_true", help="merge: widen instead of colliding live units")
    compile_parser.add_argument("--no-share", action="store_true", help="merge: do not share positively parallel units")
    compile_parser.add_argument(
        "--declare-heads",
        action="store_true",
        help="merge: give each parent hard ownership of the heads in its skills metadata",
    )
    compile_parser.add_argument("--rank", type=int, help="inject: rank-k circuit from --examples instead of rank-1")
    compile_parser.add_argument("--units", type=int, help="inject --rank: hidden-1 units in the circuit (default: spares)")
    compile_parser.add_argument("--heads", type=str, help="fold: comma-separated owned heads (default: skills metadata)")
    compile_parser.add_argument("--agreement", type=float, default=0.99, help="fold: balanced sign agreement to keep")
    compile_parser.set_defaults(func=_compile)

    args = parser.parse_args(argv)
    args.func(args)
