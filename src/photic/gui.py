from __future__ import annotations

import json
from pathlib import Path


def _hex_to_rgba(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    if len(color) != 6:
        return f"rgba(120,120,120,{alpha})"
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def run_forecast_viewer(
    forecast_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    title: str = "Photic Forecast Viewer",
) -> None:
    try:
        import plotly.graph_objects as go
        from nicegui import ui
    except ImportError as exc:
        raise ImportError(
            "Forecast viewer requires the optional GUI dependencies. "
            "Install them with `pip install nicegui plotly`."
        ) from exc

    forecast_root = Path(forecast_dir)
    if not forecast_root.exists():
        raise FileNotFoundError(f"Forecast directory does not exist: {forecast_root}")

    def list_forecast_files() -> list[Path]:
        return sorted(forecast_root.glob("*.json"))

    def load_payload(path: Path) -> dict:
        with open(path) as f:
            return json.load(f)

    def photometry_count(path: Path) -> int:
        try:
            payload = load_payload(path)
        except Exception:
            return 0
        return len(payload.get("alert_object", {}).get("photometry", []))

    def filter_forecast_files(paths: list[Path], min_points: int) -> list[Path]:
        return [path for path in paths if photometry_count(path) >= min_points]

    band_colors = {
        "u": "#7b3294",
        "g": "#1b9e77",
        "r": "#d95f02",
        "i": "#e6ab02",
        "z": "#7570b3",
        "y": "#666666",
    }

    all_forecast_files = list_forecast_files()
    min_points_state = {"value": 2}
    forecast_files = filter_forecast_files(all_forecast_files, min_points_state["value"])
    options = {str(path): path.stem for path in forecast_files}
    state = {"path": forecast_files[0] if forecast_files else None}
    file_select = None
    min_points_input = None

    def make_figure(payload: dict):
        fig = go.Figure()
        alert_obj = payload.get("alert_object", {})
        photometry = alert_obj.get("photometry", [])
        forecast = payload.get("forecast", {})
        bands = forecast.get("bands", {})

        for band, color in band_colors.items():
            points = [p for p in photometry if p.get("band") == band]
            if points:
                fig.add_trace(
                    go.Scatter(
                        x=[p["mjd"] for p in points],
                        y=[p["flux"] for p in points],
                        error_y={"type": "data", "array": [p["flux_err"] for p in points], "visible": True},
                        mode="markers",
                        marker={"color": color, "size": 7},
                        name=f"{band} photometry",
                    )
                )
            curve = bands.get(band)
            if not curve:
                continue
            x = curve.get("mjd", [])
            mu = curve.get("flux_mean", [])
            sg = curve.get("flux_sigma", [])
            if x and mu:
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=mu,
                        mode="lines",
                        line={"color": color, "width": 2},
                        name=f"{band} forecast",
                    )
                )
            if x and mu and sg:
                upper = [m + s for m, s in zip(mu, sg)]
                lower = [m - s for m, s in zip(mu, sg)]
                fig.add_trace(
                    go.Scatter(
                        x=x + list(reversed(x)),
                        y=upper + list(reversed(lower)),
                        fill="toself",
                        fillcolor=_hex_to_rgba(color, 0.12),
                        line={"color": "rgba(255,255,255,0)"},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        fig.update_layout(
            margin={"l": 40, "r": 20, "t": 30, "b": 40},
            xaxis_title="MJD",
            yaxis_title="Flux",
            legend_title="Series",
            template="plotly_white",
        )
        return fig

    @ui.refreshable
    def render(path: Path | None):
        if path is None:
            with ui.card().classes("w-full"):
                ui.label("No forecast JSON files found.")
            return
        payload = load_payload(path)
        forecast = payload.get("forecast", {})
        update = payload.get("update", {})
        alert_object = payload.get("alert_object", {})

        with ui.row().classes("w-full items-start"):
            with ui.column().classes("w-72 gap-2"):
                with ui.card().classes("w-full"):
                    ui.label(path.name).classes("text-subtitle1")
                    ui.label(f"Object ID: {forecast.get('object_id', update.get('object_id', 'unknown'))}")
                    ui.label(f"p(TDE): {forecast.get('prob_tde', float('nan')):.4f}")
                    ui.label(f"Class logit: {forecast.get('class_logit', float('nan')):.4f}")
                    ui.label(f"Context points: {forecast.get('context_points', 'n/a')}")
            with ui.column().classes("flex-1 gap-2"):
                with ui.card().classes("w-full"):
                    ui.plotly(make_figure(payload)).classes("w-full").style("height: 640px")
                with ui.card().classes("w-full"):
                    with ui.row().classes("w-full"):
                        with ui.column().classes("gap-1"):
                            ui.label("Alert Object").classes("text-subtitle2")
                            ui.label(f"Redshift: {alert_object.get('redshift', 'n/a')}")
                            ui.label(f"Photometry points: {len(alert_object.get('photometry', []))}")
                        with ui.column().classes("gap-1"):
                            ui.label("Broker Update").classes("text-subtitle2")
                            ui.label(f"First MJD: {update.get('first_mjd', 'n/a')}")
                            ui.label(f"Last MJD: {update.get('last_mjd', 'n/a')}")
                            ui.label(f"N detections: {update.get('ndet', 'n/a')}")
                        with ui.column().classes("gap-1"):
                            ui.label("Forecast").classes("text-subtitle2")
                            ui.label(f"Flux centers: {forecast.get('flux_center_by_band', 'n/a')}")
                            ui.label(f"Flux scales: {forecast.get('flux_scale_by_band', 'n/a')}")
                            ui.label(f"t_span: {forecast.get('t_span', 'n/a')}")

    def refresh_files():
        assert file_select is not None
        nonlocal all_forecast_files, forecast_files, options
        all_forecast_files = list_forecast_files()
        forecast_files = filter_forecast_files(all_forecast_files, min_points_state["value"])
        options = {str(path): path.stem for path in forecast_files}
        file_select.options = options
        if forecast_files and state["path"] not in forecast_files:
            state["path"] = forecast_files[0]
            file_select.value = str(state["path"])
        elif not forecast_files:
            state["path"] = None
            file_select.value = None
        render.refresh(state["path"])

    def on_change(_event=None):
        assert file_select is not None
        value = file_select.value
        state["path"] = Path(value) if value else None
        render.refresh(state["path"])

    def on_min_points_change(_event=None):
        assert min_points_input is not None
        min_points_state["value"] = max(int(min_points_input.value or 0), 0)
        refresh_files()

    def index():
        nonlocal file_select, min_points_input
        ui.dark_mode(False)
        ui.page_title(title)
        with ui.header().classes("items-center justify-between"):
            ui.label(title).classes("text-h5")
            ui.button("Refresh", on_click=lambda: refresh_files())
        with ui.row().classes("items-end gap-4"):
            min_points_input = ui.number(
                label="Min photometry points",
                value=min_points_state["value"],
                min=0,
                step=1,
                format="%.0f",
                on_change=on_min_points_change,
            ).classes("w-40")
            file_select = ui.select(
                options=options,
                value=str(state["path"]) if state["path"] else None,
                label="Forecast file",
                on_change=on_change,
            ).classes("w-72")
        render(state["path"])

    ui.run(host=host, port=port, title=title, reload=False, show=False, root=index)
