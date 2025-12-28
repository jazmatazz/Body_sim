"""Interactive dashboard using Plotly Dash."""

from typing import Optional, Callable

import numpy as np
from numpy.typing import NDArray

from body_sim.core.types import PressureDistribution, SimulationConfig
from body_sim.simulation.simulator import PressureSimulator


def create_plotly_heatmap(
    pressure_dist: PressureDistribution,
    title: str = "Pressure Distribution",
) -> dict:
    """Create Plotly heatmap figure data.

    Args:
        pressure_dist: Pressure distribution
        title: Plot title

    Returns:
        Plotly figure dictionary
    """
    import plotly.graph_objects as go

    fig = go.Figure(
        data=go.Heatmap(
            z=pressure_dist.grid_pressure,
            colorscale="Hot",
            colorbar=dict(title="Pressure (Pa)"),
        )
    )

    fig.update_layout(
        title=dict(
            text=f"{title}<br>Peak: {pressure_dist.peak_pressure:.0f} Pa, "
            f"Avg: {pressure_dist.average_pressure:.0f} Pa",
            x=0.5,
        ),
        xaxis_title="Column",
        yaxis_title="Row (Head → Foot)",
        yaxis=dict(autorange="reversed"),
    )

    return fig


def create_plotly_3d_surface(
    pressure_dist: PressureDistribution,
    title: str = "Pressure Surface",
) -> dict:
    """Create 3D surface plot of pressure.

    Args:
        pressure_dist: Pressure distribution
        title: Plot title

    Returns:
        Plotly figure dictionary
    """
    import plotly.graph_objects as go

    rows, cols = pressure_dist.grid_pressure.shape
    x = np.arange(cols)
    y = np.arange(rows)

    fig = go.Figure(
        data=[
            go.Surface(
                z=pressure_dist.grid_pressure,
                x=x,
                y=y,
                colorscale="Hot",
                colorbar=dict(title="Pressure (Pa)"),
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Column",
            yaxis_title="Row",
            zaxis_title="Pressure (Pa)",
        ),
    )

    return fig


def create_plotly_time_series(
    history: list[PressureDistribution],
    title: str = "Pressure Over Time",
) -> dict:
    """Create time series plot.

    Args:
        history: List of pressure distributions
        title: Plot title

    Returns:
        Plotly figure dictionary
    """
    import plotly.graph_objects as go

    times = [pd.timestamp for pd in history]
    peaks = [pd.peak_pressure for pd in history]
    avgs = [pd.average_pressure for pd in history]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=times, y=peaks, mode="lines", name="Peak Pressure", line=dict(color="red"))
    )
    fig.add_trace(
        go.Scatter(
            x=times, y=avgs, mode="lines", name="Average Pressure", line=dict(color="blue")
        )
    )

    # Add threshold line
    fig.add_hline(
        y=4266, line_dash="dash", line_color="orange", annotation_text="Capillary Threshold"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Pressure (Pa)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


class SimulationDashboard:
    """Interactive dashboard for simulation visualization.

    Uses Plotly Dash for web-based interactive visualization.
    """

    def __init__(self, simulator: PressureSimulator, port: int = 8050):
        """Initialize dashboard.

        Args:
            simulator: Pressure simulator instance
            port: Port for web server
        """
        self.simulator = simulator
        self.port = port
        self._app = None

    def create_app(self):
        """Create Dash application."""
        try:
            from dash import Dash, html, dcc, callback, Output, Input, State
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "Dash is required for the dashboard. Install with: pip install dash"
            )

        app = Dash(__name__)

        app.layout = html.Div(
            [
                html.H1("Pressure Simulation Dashboard", style={"textAlign": "center"}),
                html.Div(
                    [
                        # Controls
                        html.Div(
                            [
                                html.H3("Controls"),
                                html.Label("Grid Rows:"),
                                dcc.Slider(
                                    id="grid-rows",
                                    min=4,
                                    max=64,
                                    step=4,
                                    value=self.simulator.config.grid_rows,
                                    marks={i: str(i) for i in range(4, 65, 8)},
                                ),
                                html.Label("Grid Columns:"),
                                dcc.Slider(
                                    id="grid-cols",
                                    min=4,
                                    max=64,
                                    step=4,
                                    value=self.simulator.config.grid_cols,
                                    marks={i: str(i) for i in range(4, 65, 8)},
                                ),
                                html.Label("Body Weight (kg):"),
                                dcc.Slider(
                                    id="body-weight",
                                    min=40,
                                    max=150,
                                    step=5,
                                    value=self.simulator.config.body_weight,
                                    marks={i: str(i) for i in range(40, 151, 20)},
                                ),
                                html.Label("Pattern:"),
                                dcc.Dropdown(
                                    id="pattern-select",
                                    options=[
                                        {"label": "Alternating", "value": "alternating"},
                                        {"label": "Wave", "value": "wave"},
                                        {"label": "Static", "value": "static"},
                                    ],
                                    value="alternating",
                                ),
                                html.Br(),
                                html.Button("Run Step", id="run-step", n_clicks=0),
                                html.Button(
                                    "Run 10 Steps", id="run-10-steps", n_clicks=0
                                ),
                                html.Button("Reset", id="reset", n_clicks=0),
                            ],
                            style={"width": "25%", "display": "inline-block", "verticalAlign": "top"},
                        ),
                        # Visualization
                        html.Div(
                            [
                                dcc.Tabs(
                                    [
                                        dcc.Tab(
                                            label="Pressure Heatmap",
                                            children=[dcc.Graph(id="heatmap")],
                                        ),
                                        dcc.Tab(
                                            label="3D Surface",
                                            children=[dcc.Graph(id="surface-3d")],
                                        ),
                                        dcc.Tab(
                                            label="Time Series",
                                            children=[dcc.Graph(id="time-series")],
                                        ),
                                    ]
                                ),
                                html.Div(id="metrics-display"),
                            ],
                            style={"width": "70%", "display": "inline-block"},
                        ),
                    ]
                ),
                # Store for simulation state
                dcc.Store(id="simulation-state"),
            ]
        )

        @app.callback(
            [
                Output("heatmap", "figure"),
                Output("surface-3d", "figure"),
                Output("time-series", "figure"),
                Output("metrics-display", "children"),
            ],
            [
                Input("run-step", "n_clicks"),
                Input("run-10-steps", "n_clicks"),
                Input("reset", "n_clicks"),
            ],
            [
                State("pattern-select", "value"),
                State("body-weight", "value"),
            ],
        )
        def update_visualization(
            step_clicks, ten_steps_clicks, reset_clicks, pattern_type, body_weight
        ):
            from dash import ctx

            # Handle reset
            if ctx.triggered_id == "reset":
                self.simulator.reset()

            # Update body weight if changed
            if body_weight != self.simulator.config.body_weight:
                self.simulator.set_body_weight(body_weight)

            # Update pattern if changed
            from body_sim.mattress.patterns import create_pattern

            new_pattern = create_pattern(pattern_type)
            self.simulator.set_pattern(new_pattern)

            # Run steps
            if ctx.triggered_id == "run-step":
                if self.simulator.current_mesh is not None:
                    self.simulator.step()
            elif ctx.triggered_id == "run-10-steps":
                if self.simulator.current_mesh is not None:
                    self.simulator.run(self.simulator.config.time_step * 10)

            # Get current state
            history = self.simulator.history
            if history:
                current_pressure = history[-1]
            else:
                # Create empty pressure distribution
                current_pressure = PressureDistribution(
                    grid_pressure=np.zeros(
                        (self.simulator.config.grid_rows, self.simulator.config.grid_cols),
                        dtype=np.float32,
                    ),
                    peak_pressure=0.0,
                    average_pressure=0.0,
                    contact_area=0.0,
                    timestamp=0.0,
                )

            # Create figures
            heatmap_fig = create_plotly_heatmap(current_pressure)
            surface_fig = create_plotly_3d_surface(current_pressure)
            time_fig = create_plotly_time_series(history) if history else go.Figure()

            # Metrics display
            summary = self.simulator.get_pressure_summary()
            metrics = html.Div(
                [
                    html.H4("Simulation Metrics"),
                    html.P(f"Time: {self.simulator.simulation_time:.1f} s"),
                    html.P(f"Steps: {summary['num_steps']}"),
                    html.P(f"Max Peak Pressure: {summary['max_peak_pressure']:.0f} Pa"),
                    html.P(f"Avg Peak Pressure: {summary['avg_peak_pressure']:.0f} Pa"),
                    html.P(f"Avg Contact Area: {summary['avg_contact_area']:.1%}"),
                ]
            )

            return heatmap_fig, surface_fig, time_fig, metrics

        self._app = app
        return app

    def run(self, debug: bool = False) -> None:
        """Run the dashboard server.

        Args:
            debug: Enable debug mode
        """
        if self._app is None:
            self.create_app()

        self._app.run_server(debug=debug, port=self.port)

    def update(self, pressure_dist: PressureDistribution) -> None:
        """Update callback for simulation steps.

        Args:
            pressure_dist: New pressure distribution
        """
        # This method can be used as a callback during simulation.run()
        # In a real implementation, this would update the dashboard state
        pass
