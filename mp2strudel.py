"""

MP2Strudel workflow:

1. Pull in data from Materials Project API
2. Fit the phonon DOS with Gaussian peaks (SingingMaterials package)
3. Push processed data as a JSON file to GitHub (data/<formula>_<mpid>_<projection>_<temp>.json, e.g. CaCO3_mp3953_element_300K.json)
4. Print Strudel code that loads the JSON, to paste at the top of the `strudel.cc` editor.

Example usage:

    python mp2strudel.py mp-66 mp-3953 --temps athermal 300 600
    python mp2strudel.py mp-3953 --n-peaks Ca=6 C=4 O=6 --smoothing Ca=0.1

Environment:

This code must be run from inside a clone of a public GitHub repo, with MP_API_KEY in a .env file:

Options:
    --temps          "athermal" and/or temperatures in K (default: athermal)
    --projection     total, element or site (default: element)
    --n-peaks        one number for every projection, or LABEL=N pairs (default 9)
    --smoothing      THz; one number, or LABEL=X pairs (default 0.3)
    --min-frequency  THz; leave the lowest frequencies out of the fit (default 0)
    --octaves-down   N; play the true phonon frequencies N octaves lower, i.e. divide by 2**N
                     (default 33, which puts 1 THz at 116.4 Hz)
    --pitch-map      THZ HZ; instead of --octaves-down, phonon frequency THZ sounds at HZ and every other
                     frequency is scaled by the same factor (e.g. 1 110 puts 1 THz on A2)
    --png            also save a plot of each fit to plots/<same name>.png (committed with the data)
    --no-push        write the files but skip git
    --copy           also copy the Strudel block to the clipboard (macOS pbcopy)
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from singingmaterials import frequency_mapping, mp_interface, peak_fitting

DATA_DIR = Path("data")
PLOT_DIR = Path("plots")
DEFAULT_OCTAVES_DOWN = frequency_mapping.DEFAULT_OCTAVES_DOWN
DEFAULT_PITCH_MAP = (  # (THz, Hz): 1 THz sounds at 116.4 Hz
    1.0,
    frequency_mapping.phonon_to_audible_octaves(1e12, DEFAULT_OCTAVES_DOWN),
)
AUDIBLE_RANGE = (
    20.0,
    20000.0,
)  # Hz, typical limits of human hearing - used in warnings
COMFORTABLE_RANGE = (
    50.0,
    5000.0,
)  # Hz, where pitch is clear and most speakers are audible - used in warnings


def temp_tag(temp):
    """'athermal', '300K', or '300p5K' for 300.5 K ('.' is not allowed in a JS name)."""
    return "athermal" if temp is None else f"{temp}K".replace(".", "p")


def export_name(mp_id, formula, projection, temp):
    """one name used both as the file stem and as the JS const in the Strudel block,
    e.g. CaCO3_mp3953_element_300K."""
    return f"{formula}_{mp_id.replace('-', '')}_{projection}_{temp_tag(temp)}"


def settings_for_material(setting, labels):
    """keep only the specified LABEL=value pairs."""
    if isinstance(setting, dict):
        return {k: v for k, v in setting.items() if k in labels}
    else:
        return setting


def pitch_warnings(peaks):
    """one line per peak whose audible pitch is outside the comfortable range, or outside range of typical human hearing."""
    lines = []
    for label, v in peaks.items():
        for f, a in zip(v["frequency"], v["audible"]):
            if AUDIBLE_RANGE[0] <= a <= AUDIBLE_RANGE[1]:
                if COMFORTABLE_RANGE[0] <= a <= COMFORTABLE_RANGE[1]:
                    continue
                why = (
                    "low, may be inaudible on small speakers"
                    if a < COMFORTABLE_RANGE[0]
                    else "high, may sound shrill"
                )
            else:
                why = (
                    "below human hearing"
                    if a < AUDIBLE_RANGE[0]
                    else "above human hearing"
                )
            lines.append(f"{label} peak at {f / 1e12:.2f} THz -> {a:g} Hz ({why})")
    return lines


def peaks_to_json(fits, pitch_map=DEFAULT_PITCH_MAP):
    """one entry per label (element, site or 'total'), each holding lists ordered by frequency.
    DOS ratios are relative to the tallest peak across all labels, as in peak_fitting.save_peak_fit."""
    all_dos = np.concatenate([f["dos"] for f in fits.values()])
    ref_dos = np.max(all_dos) if len(all_dos) else 1.0
    dos_sum = np.sum(all_dos / ref_dos) if len(all_dos) else 1.0

    def r(values, n=6):
        return [float(round(float(v), n)) for v in values]

    out = {}
    for label, f in fits.items():
        out[label] = {
            "n": len(f["centres"]),
            "frequency": [float(c) for c in f["centres"]],  # Hz
            "fwhm": [float(w) for w in f["fwhms"]],  # Hz
            "dos": r(f["dos"]),
            "dos_ratio": r(f["dos"] / ref_dos, 4),  # dos / tallest peak
            "norm_dos_ratio": r(
                f["dos"] / ref_dos / dos_sum, 4
            ),  # sums to 1 over all labels
            "audible": r(
                frequency_mapping.phonon_to_audible_scaled(
                    f["centres"], pitch_map[0] * 1e12, pitch_map[1]
                ),
                2,
            ),  # Hz, frequency scaled down by the pitch map (ratios kept)
        }
    return out


def save_plot(fits, formula, path):
    """save peak_fitting fits as a png using SingingMaterials' plot_peak_fits."""
    import matplotlib

    matplotlib.use("Agg")  # draw to file without opening windows
    import matplotlib.pyplot as plt
    from singingmaterials import visualisation

    axes = visualisation.plot_peak_fits(fits)
    fig = axes[0].figure
    first = next(iter(fits.values()))
    at = "" if first["temperature"] is None else f" at {first['temperature']:g} K"
    fig.suptitle(
        f"{formula} ({first['mp_id']}): phonon DOS peaks by {first['projection']}{at}"
    )
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")


def export_material(
    mp_id,
    temps,
    projection,
    n_peaks,
    smoothing,
    min_frequency,
    pitch_map=DEFAULT_PITCH_MAP,
    octaves_down=DEFAULT_OCTAVES_DOWN,
    png=False,
):
    """fit and write one JSON file per temperature; return a description of what was written."""
    dos_dict = mp_interface.get_dos_raw_mp(mp_id)  # cached locally as <mp_id>_dos.json
    formula = dos_dict["metadata"]["formula"]
    labels = list(peak_fitting._projections(dos_dict, projection))
    n_peaks_m = settings_for_material(n_peaks, labels)
    smoothing_m = settings_for_material(smoothing, labels)

    files, plots = {}, []
    for temp in temps:
        fits = peak_fitting.fit_dos_peaks(
            dos_dict=dos_dict,
            projection=projection,
            temperature=temp,
            n_peaks=n_peaks_m,
            smoothing=smoothing_m,
            min_frequency=min_frequency,
        )
        payload = {
            "mp_id": mp_id,
            "formula": formula,
            "temperature": temp,
            "projection": projection,
            "settings": {
                "n_peaks": n_peaks_m,
                "smoothing_THz": smoothing_m,
                "min_frequency_THz": min_frequency,
                "pitch_map": {"THz": pitch_map[0], "Hz": pitch_map[1]},
                "octaves_down": octaves_down,  # None when --pitch-map was used
            },
            "peaks": peaks_to_json(fits, pitch_map),
        }
        path = DATA_DIR / f"{export_name(mp_id, formula, projection, temp)}.json"
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
        counts = ", ".join(f"{k}:{v['n']}" for k, v in payload["peaks"].items())
        print(f"wrote {path}  (peaks {counts})")
        warnings = pitch_warnings(payload["peaks"])
        if warnings:
            print(
                f"  warning: {len(warnings)} peak(s) outside comfortable hearing "
                f"({COMFORTABLE_RANGE[0]:g}-{COMFORTABLE_RANGE[1]:g} Hz); "
                "adjust --octaves-down or --pitch-map if needed:"
            )
            for line in warnings:
                print(f"    {line}")
        files[temp] = path
        if png:
            plot_path = PLOT_DIR / path.with_suffix(".png").name
            save_plot(fits, formula, plot_path)
            plots.append(plot_path)

    return {
        "mp_id": mp_id,
        "formula": formula,
        "labels": labels,
        "projection": projection,
        "files": files,
        "plots": plots,
    }


def git(*args):
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def github_repo():
    """'owner/repo' from the origin remote (https or ssh form)."""
    url = git("remote", "get-url", "origin")
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?/?$", url)
    if not match:
        raise SystemExit(f"origin ({url}) is not a GitHub repo")
    return match.group(1)


def commit_and_push(paths, message):
    git("add", *map(str, paths))
    if git("diff", "--cached", "--name-only"):
        git("commit", "-m", message)
        print(git("log", "-1", "--oneline"))
    else:
        print("data unchanged, nothing to commit")
    print("pushing...")
    subprocess.run(["git", "push"], check=True)
    return git("rev-parse", "HEAD")


def strudel_block(materials, base_url, label):
    lines = [
        f"// Singing Materials phonon peaks · {label}",
        "const load = f => fetch('" + base_url + "' + f).then(r => r.json())",
    ]
    for m in materials:
        lines.append("")
        lines.append(
            f"// {m['formula']} ({m['mp_id']}) · {m['projection']} peaks: {', '.join(m['labels'])}"
        )
        for _, path in m["files"].items():
            lines.append(f"const {path.stem} = await load('{path.name}')")
    first = materials[0]
    ex = f"{next(iter(first['files'].values())).stem}.peaks.{first['labels'][0]}"
    lines += [
        "",
        f"// {ex}.frequency   peak centres (Hz)    .fwhm  widths (Hz)",
        f"// {ex}.audible     peak centres scaled to audible pitches (Hz)",
        f"// {ex}.dos_ratio   dos / tallest peak    .modes  modes per peak",
    ]
    return "\n".join(lines)


def parse_temp(value):
    if value.lower() in ("athermal", "none"):
        return None
    t = float(value)
    if t <= 0:
        raise argparse.ArgumentTypeError(
            "temperatures must be above 0 K (use 'athermal' instead)"
        )
    return int(t) if t.is_integer() else t


def parse_setting(values, cast):
    """['6'] -> 6;  ['Ca=6', 'C=4'] -> {'Ca': 6, 'C': 4};  None -> None"""
    if values is None:
        return None
    if len(values) == 1 and "=" not in values[0]:
        return cast(values[0])
    try:
        return {k: cast(v) for k, v in (item.split("=", 1) for item in values)}
    except ValueError:
        raise SystemExit(
            f"expected one number or LABEL=value pairs, got {' '.join(values)}"
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("mp_ids", nargs="+")
    p.add_argument("--temps", nargs="+", type=parse_temp, default=[None], metavar="T")
    p.add_argument(
        "--projection", choices=["total", "element", "site"], default="element"
    )
    p.add_argument("--n-peaks", nargs="+", metavar="N")
    p.add_argument("--smoothing", nargs="+", metavar="THZ")
    p.add_argument("--min-frequency", type=float, default=0.0, metavar="THZ")
    pitch = p.add_mutually_exclusive_group()
    pitch.add_argument(
        "--octaves-down", type=int, default=DEFAULT_OCTAVES_DOWN, metavar="N"
    )
    pitch.add_argument("--pitch-map", nargs=2, type=float, metavar=("THZ", "HZ"))
    p.add_argument("--png", action="store_true")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--copy", action="store_true")
    args = p.parse_args()

    if args.pitch_map is not None:
        if min(args.pitch_map) <= 0:
            p.error("--pitch-map needs two positive numbers: THZ HZ")
        pitch_map, octaves_down = tuple(args.pitch_map), None
    else:
        octaves_down = args.octaves_down
        pitch_map = (
            1.0,
            frequency_mapping.phonon_to_audible_octaves(1e12, octaves_down),
        )
    temps = list(dict.fromkeys(args.temps))  # drop repeats, keep order
    n_peaks = parse_setting(args.n_peaks, int) or peak_fitting.DEFAULT_N_PEAKS
    smoothing = parse_setting(args.smoothing, float) or peak_fitting.DEFAULT_SMOOTHING

    DATA_DIR.mkdir(exist_ok=True)
    if args.png:
        PLOT_DIR.mkdir(exist_ok=True)
    materials = [
        export_material(
            mp_id,
            temps,
            args.projection,
            n_peaks,
            smoothing,
            args.min_frequency,
            pitch_map,
            octaves_down,
            args.png,
        )
        for mp_id in args.mp_ids
    ]

    # index of everything exported so far (all runs), so you can see what's available
    index = []
    for f in sorted(p for p in DATA_DIR.glob("*.json") if p.name != "index.json"):
        d = json.loads(f.read_text())
        index.append(
            {
                "file": f.name,
                "mp_id": d["mp_id"],
                "temperature": d["temperature"],
                "projection": d.get("projection"),
                "labels": list(d.get("peaks", {})),
            }
        )
    (DATA_DIR / "index.json").write_text(
        json.dumps(index, indent=1, ensure_ascii=False)
    )

    paths = (
        [path for m in materials for path in m["files"].values()]
        + [DATA_DIR / "index.json"]
        + [path for m in materials for path in m["plots"]]
    )
    if args.no_push:
        repo, ref = "OWNER/REPO", "main"
        try:
            repo, ref = github_repo(), git("rev-parse", "--abbrev-ref", "HEAD")
        except (subprocess.CalledProcessError, SystemExit, FileNotFoundError):
            pass
        print("\n--no-push: these links will only work once the files are pushed")
    else:
        repo = github_repo()
        mats = ", ".join(f"{m['formula']} {m['mp_id']}" for m in materials)
        ref = commit_and_push(
            paths, f"phonon peaks: {mats} ({', '.join(map(temp_tag, temps))})"
        )

    # links pinned to the commit so we never get a stale cached copy
    base_url = f"https://raw.githubusercontent.com/{repo}/{ref}/{DATA_DIR.as_posix()}/"
    block = strudel_block(materials, base_url, f"{repo}@{ref[:7]}")

    print("\n---------- paste into the top of strudel.cc ----------\n")
    print(block)
    print("\n-------------------------------------------------------")

    if args.copy:
        subprocess.run(["pbcopy"], input=block, text=True, check=True)
        print("(copied to clipboard)")


if __name__ == "__main__":
    main()
