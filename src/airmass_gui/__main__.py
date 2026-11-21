"""Entry point: airmass-gui [--no-gui] [--out FILE] [--site [OUT]]."""
from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="airmass-gui", description="Airmass charts for astronomical targets.")
    parser.add_argument("--no-gui", action="store_true", help="render a PNG from the saved workspace (no display needed)")
    parser.add_argument("--out", default=None, help="output PNG path for --no-gui")
    parser.add_argument("--site", nargs="?", const="site/index.html", default=None, metavar="OUT",
                        help="write an interactive single-file HTML page from the saved workspace "
                             "(default: site/index.html)")
    args = parser.parse_args(argv)

    if args.site is not None:
        from pathlib import Path

        from .site import build_site
        from .workspace import default_path, load, new_workspace, save

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
