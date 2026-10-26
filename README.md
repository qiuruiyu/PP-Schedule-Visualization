# Pipeline Parallelism Emulation and Visualization

This project provides tools for emulating and visualizing pipeline parallelism strategies used in large language model training.

It computes operation start and end times from scheduling dependencies and the supplied compute and communication timings. The results are simulations, not measurements from a training run.

## Overview

Pipeline parallelism is a technique used to train large models by partitioning the model across multiple devices and processing data in a pipelined fashion. This project allows you to:

- Simulate different pipeline parallelism strategies (1F1B, Interleaved, Zero-Bubble, etc.)
- Visualize the execution schedule on multiple devices
- Compare different strategies for efficiency

## Features

- **Supported Pipeline Strategies**:
  - 1F1B (One-Forward-One-Backward)
  - Interleaved 1F1B
  - Zero-Bubble 1F1B (ZB-1P)
  - 1F1B with overlapped forward and backward computation
  - Interleaved 1F1B with overlapped forward and backward computation
  - DualPipe (Bidirectional pipeline parallelism with full forward-backward overlap)
  - DualPipeV (Virtual stages with zero-bubble backward scheduling)

- **Visualization**:
  - Interactive visualization dashboard using Plotly/Dash
  - Multiple strategy selection, tabs with a shared time scale, and total execution time comparison
  - Hover details for each operation: microbatch, stage, type, start, end, and duration
  
- **Configuration**:
  - Configurable simulation parameters through Hydra
  - Customizable stage latency and communication costs

## Installation

This project requires Python 3.10 or newer and uses [uv](https://github.com/astral-sh/uv) for dependency management.

Install uv using its [installation instructions](https://docs.astral.sh/uv/getting-started/installation/), then clone this repository:

```bash
git clone https://github.com/qiuruiyu/PP-Schedule-Visualization.git
cd PP-Schedule-Visualization
uv sync
```


## Running the Interactive Server

To visualize schedules interactively:

```bash
uv run app.py
```

Open `http://127.0.0.1:7860/` in your browser. The web app listens on `0.0.0.0:7860`.

1. Set the device count, total stage count, and number of microbatches. Microbatch count means the number of small batches, not the number of samples in each batch.
2. Enter P2P latency and operation timings in milliseconds.
3. Select one or more strategy cards, then click **Generate Schedules**. Click it again after changing parameters or strategies.
4. Switch between the result tabs and compare total execution times below the chart. **Relative Efficiency** is the shortest time among the selected strategies divided by each strategy's time; it is not GPU utilization.

### Web Timing Inputs

The web UI treats compute timings as the total work assigned to one device for one microbatch. It divides these timings by the number of stages per device before simulation. For example, with 4 devices and 8 interleaved stages, forward = 1 ms and backward = 2 ms become 0.5 ms and 1 ms per stage. P2P latency is not divided.

Under **Customize Per-Stage Timing**, individual forward and backward entries override the corresponding global values; empty entries use the global value. These entries use the same scaling as the global timings. For strategies with split backward computation, the global **Backward D** and **Backward W** values determine backward timings.

Under **Advanced Timing Options**:

- **Backward D / Backward W** specify input-gradient and weight-gradient computation times. Both are required for ZB-1P, DualPipe, and DualPipeV. When comparing against ordinary backward computation, keep their sum equal to the combined backward time.
- **Overlapped Forward+Backward Time** specifies the combined duration of an overlapping block. If left empty, the simulator uses the sum of its component operation times.
- **Microbatch Group Size per VP Stage** controls interleave grouping; leaving it empty uses the device count.

### Stage Counts

| Web strategy | Stage count used by the web UI |
| --- | --- |
| 1F1B, ZB-1P, 1F1B Overlap | Equal to the device count; adjusted automatically |
| 1F1B Interleave, 1F1B Interleave + Overlap | Must be divisible by the device count |
| DualPipe | Equal to the device count; adjusted automatically; requires an even device count |
| DualPipeV | Twice the device count; adjusted automatically; requires an even device count |

The UI displays a notification when it adjusts stage counts or scales timings. A strategy may also impose constraints on microbatch count; any simulation error is shown in a notification.

For a first comparison, use 4 devices, 8 stages, 8 microbatches, P2P = 0 ms, forward = 1 ms, and backward = 2 ms. Select **1F1B** and **1F1B Interleave**. Their total times are 33 ms and 28.5 ms respectively; the UI uses 4 stages for 1F1B and 8 for Interleave.

## Running from Command Line

`main.py` runs a single strategy and opens its Dash visualization. Its default port is **8050**, set by `visualization_port` in `conf/config.yaml`.

Unlike the web UI, CLI operation timings are **per stage** and are passed directly to the simulator. The CLI does not automatically adjust stage counts or divide timings by stages per device. To reproduce the web comparison above for Interleave, use:

```bash
uv run python main.py strategy=interleave num_devices=4 num_stages=8 num_batches=8 op_times.forward=0.5 op_times.backward=1.0
```

### Running for 1F1B strategy:
```bash
uv run python main.py strategy=1f1b num_devices=4 num_stages=4 num_batches=8
```
![1f1b](assets/1f1b.png)

### Running for interleaved strategy:
```bash
uv run python main.py strategy=interleave num_devices=4 num_stages=8 num_batches=8
```
![interleave](assets/interleave_1f1b.png)

You can optionally set `microbatch_group_size_per_vp_stage` for the `interleave` strategy.

### Running for ZB-1P strategy:
```bash
uv run python main.py strategy=zb1p num_devices=4 num_stages=4 num_batches=8
```
![zb1p](assets/zb1p.png)

### Running for DualPipe strategy:
```bash
uv run python main.py strategy=dualpipe num_devices=8 num_stages=8 num_batches=20
```
![dualpipe](assets/dualpipe.png)

### Running for DualPipe-V strategy
```bash
uv run python main.py strategy=dualpipe_v num_devices=4 num_stages=8 num_batches=10
```
![dualpipe_v](assets/dualpipe_v.png)

### Running for 1F1B-batch-overlap strategy:
```bash
uv run python main.py strategy=1f1b_overlap num_devices=4 num_stages=4 num_batches=8
```
![1f1b_overlap](assets/1f1b_overlap.png)

### Running for 1F1B-interleave-overlap strategy:
```bash
uv run python main.py strategy=1f1b_interleave_overlap num_devices=4 num_stages=8 num_batches=8
```
![1f1b_interleave_overlap](assets/1f1b_interleave_overlap.png)


## Configuration

The default configuration is in `conf/config.yaml`. You can override any parameter on the command line or create configuration groups for different scenarios.

#### Override Specific Parameters

You can override specific parameters at runtime:
```bash
uv run python main.py op_times.forward=0.5 op_times.backward=1.0 num_batches=6
```

Use DualPipe as an example, you can manually set different time for forward/backward/backward_D/backward_W/overlapped_forward_backward:
```bash
uv run python main.py strategy=dualpipe num_devices=8 num_stages=8 num_batches=32 op_times.forward=1.0 op_times.backward=2.0 op_times.backward_D=1.0 op_times.backward_W=1.0 op_times.overlapped_forward_backward=2.5
```


### Using Different Configuration Files

You can use different configuration files with Hydra in several ways:

#### Recommended Approach

1. Create multiple configuration files in the `conf` directory for different use cases:
   ```
   conf/
   ├── config.yaml     # Default configuration
   └── model_A.yaml    # Example configuration with stage-specific timings
   ```

2. Run with your desired configuration using the `--config-name` flag:
   ```bash
   uv run python main.py --config-name=model_A
   ```


## Project Structure

```
PP-Schedule-Visualization/
├── app.py                  # Multi-strategy web UI (port 7860)
├── assets/                 # CSS, client-side interactions, and screenshots
├── conf/                   # Hydra configuration files
│   └── config.yaml         # Default configuration
├── src/                    # Source code
│   ├── __init__.py         # Package initialization
│   ├── execution_model.py  # Schedule execution models
│   ├── strategies.py       # Pipeline parallelism strategies
│   └── visualizer.py       # Visualization utilities
├── main.py                 # Single-strategy CLI and visualization
├── tests/                  # Simulator and web callback tests
├── examples/               # Plotting examples, including Megatron-LM timer data
├── formula.py              # Analytical timing checks against simulation
├── pyproject.toml          # Project metadata and dependencies
└── README.md               # This file
```

## Development

Run the tests with the development dependencies:

```bash
uv run --extra dev python -m pytest tests/ -q
```

To compare analytical timing formulas with simulation results:

```bash
uv run python formula.py
```

Dependencies are maintained in `pyproject.toml`; `requirements.txt` is used by the Docker deployment. Update both when changing dependencies. The deployment entry point is `app:server`, listening on port 7860 through Gunicorn.

For plotting timers collected from a real Megatron-LM run, see [examples/megatron-lm/README.md](examples/megatron-lm/README.md).

## References

1. _PipeDream: Fast and Efficient Pipeline Parallel DNN Training_. [arxiv](https://arxiv.org/abs/1806.03377)
2. _Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM_. [arxiv](https://arxiv.org/abs/2104.04473)
3. _Zero Bubble Pipeline Parallelism_. [arxiv](https://arxiv.org/abs/2401.10241)
4. _Communication-Computation Overlap in MoE Training with 1F1B Pipeline Parallelism_. [blog](https://zhuanlan.zhihu.com/p/28463368206)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
