"""Entry point: airmass-gui [--no-gui] [--out FILE] [--site [OUT]] [--workspace FILE]."""
from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    """Run the airmass-gui CLI.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments after the program name; sys.argv is used
        when omitted.

    Returns
    -------
    None
        With --site, writes the HTML page to the --site path from the
        saved workspace, or from --workspace FILE when given. With
        --no-gui, writes a PNG to --out (default: airmass_chart.png next
        to the workspace). Otherwise the tkinter app runs until closed.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="airmass-gui", description="Airmass charts for astronomical targets.")
    parser.add_argument("--no-gui", action="store_true", help="render a PNG from the saved workspace (no display needed)")
    parser.add_argument("--out", default=None, help="output PNG path for --no-gui")
    parser.add_argument("--site", nargs="?", const="site/index.html", default=None, metavar="OUT",
                        help="write an interactive single-file HTML page from the saved workspace "
                             "(default: site/index.html)")
    parser.add_argument("--workspace", default=None, metavar="FILE",
                        help="workspace JSON to embed with --site "
                             "(default: the saved workspace)")
    args = parser.parse_args(argv)

    if args.site is not None:
        from pathlib import Path

        from .site import build_site
        from .workspace import default_path, load, new_workspace, save

        if args.workspace is not None:
            ws = Path(args.workspace)
            if not ws.is_file():
                raise SystemExit("airmass-gui: workspace not found: %s" % ws)
            data = load(ws)
        else:
            path = default_path()
            if not path.exists():
                save(path, new_workspace())
            data = load(path)
        out = build_site(data, args.site)
        print(f"Wrote {out} (observatory: {data['observatory'].get('name')})")
        return

    if args.no_gui:
        from pathlib import Path

        from .chart import render_figure
        from .workspace import default_path, load, new_workspace, save

        path = default_path()
        if not path.exists():
            save(path, new_workspace())
        data = load(path)
        out = Path(args.out) if args.out else path.with_name("airmass_chart.png")
        render_figure(data, out)
        print(f"Wrote {out} (observatory: {data['observatory'].get('name')})")
        return

    from .app import App

    App().run()


if __name__ == "__main__":
    main()
