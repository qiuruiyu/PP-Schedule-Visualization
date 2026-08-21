# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Pipeline Parallelism（流水线并行）调度策略的**模拟器 + 可视化工具**。它不做真实训练，而是给定每个 stage 的算子耗时，用依赖图推导出各 device 上算子的 start/end 时间，再用 Plotly 画成甘特图。

## 常用命令

依赖用 [uv](https://github.com/astral-sh/uv) 管理，以 `pyproject.toml` 为准（`uv.lock` 被 gitignore，未入库）；`requirements.txt` 只服务于 Dockerfile / HF Spaces 部署，改依赖时两处都要同步。

```bash
# 交互式 Web UI（本地开发主入口，监听 0.0.0.0:7860；README 里写的 8050 已过时）
uv run app.py

# CLI 单策略模拟 + 起 Dash 服务（Hydra 配置，见 conf/config.yaml）
uv run python main.py strategy=1f1b num_devices=4 num_stages=4 num_batches=8
uv run python main.py strategy=dualpipe_v num_devices=4 num_stages=8 num_batches=10
uv run python main.py op_times.forward=0.5 op_times.backward=1.0   # 覆盖任意参数
uv run python main.py --config-name=model_A                        # 换配置文件

# 测试（pytest 在 dev extra 里，不加 --extra dev 会报 program not found）
uv run --extra dev pytest tests/ -q
uv run --extra dev pytest tests/test_app_dualpipe_v.py::test_dualpipe_v_is_available_to_the_web_ui -q

# 公式验证脚本：对比解析公式与模拟器算出的总时间
uv run python formula.py

# 从 Megatron-LM dump 的真实 timer 数据画图（用法见 examples/megatron-lm/README.md）
uv run examples/megatron-lm/plot.py --data-dir $DATA_DIR --pp-size 4 --vpp-size 1
```

代码风格工具已在 `pyproject.toml` 配置好（black line-length 88、isort black profile、mypy strict-ish），但现有代码并未全部符合 mypy 的 `disallow_untyped_defs`——不要为了过 mypy 去大改无关文件。

## 架构

三层，依赖方向单向：`strategies` → `execution_model` ← `visualizer`。

### 1. `src/execution_model.py`：调度的数据模型与执行引擎

- **`Operation`**：一个 `(batch_id, stage_id, op_type)` 三元组即唯一 key。`op_type` ∈ `forward` / `backward` / `backward_D`（算输入梯度）/ `backward_W`（算权重梯度）。`device_id` 由入队时赋值，**重复入队会 assert 失败**。
- **`OverlappedOperation`**：把同一 device 上几个 op 打包成一个"并发执行"的块，`op_type` 拼成 `overlapped_forward_backward`。它同时以自身和各子 op 的 key 注册进 `Schedule.ops`，`get_op()` 会透明地返回外层块——这是 overlap 类策略能复用普通依赖推导的关键。
- **`ScheduleConfig`**：所有参数的唯一来源。两个易踩的地方：
  - `op_times` 的每个值可以是 `float`（全 stage 相同）或 `Dict[stage_id, float]`（逐 stage 指定），`get_op_time()` 统一处理；`overlapped_*` 若没显式给值，会 fallback 成两个子 op 时间之和。
  - `placement_strategy` 决定 `device_to_stages` 映射：`standard`（连续切分）/ `interleave`（`i % num_devices`）/ `dualpipe`、`dualpipe_v`（每个 device 拿 `[i, num_stages-1-i]` 一头一尾两个 stage，即 V 形放置）。各策略的 `num_devices`/`num_stages` 约束在 `init_device_to_stages()` 里以 assert 表达。
- **`Schedule.get_dependencies()`**：整个模拟的语义核心。每个 op 依赖 = P2P 依赖（forward 依赖上一 stage 同 batch 的 forward；backward 系依赖下一 stage）+ device 内的前一个 op。`p2p_latency > 0` 时还有一段 **sync P2P** 逻辑：仅当前后两个 op 都非 overlapped、base type 相同且 stage 相同（即 warmup/cooldown 的纯 forward 或纯 backward 串）时，才在 device 依赖上额外加一个 `p2p_latency` 间隙。改这段要特别小心，条件放宽会在 interleave 调度里制造依赖环。
- **`Schedule.execute()`**：按 device 队列下标逐轮推进 + 递归解依赖（非拓扑排序，代码里有 TODO）。之后可取 `get_total_execution_time()` / `get_bubble_rate()` / `get_device_running_time()`。

### 2. `src/strategies.py`：各调度策略

每个 `generate_*_schedule(config) -> Schedule` 只做一件事：**决定每个 device 队列里 op 的先后顺序**，不碰时间。时间完全由 `execute()` 推出来。

- 简单策略（`1f1b`、`zb1p`、`1f1b_overlap`）：`Schedule(config)` 预建全部 op，然后 warmup / steady / cooldown 三段循环 `add_operation`。
- interleave 系：`_get_pp_rank_microbatches` / `_get_schedule_table` / `_convert_schedule_table_to_order` 三个 helper 是从 Megatron-LM 移植的，产出一个正负号表示 forward/backward 的 `order` 列表，再翻译成 op 入队。**改这里前先确认与 Megatron 的行为是否要保持一致。**
- dualpipe 系：用 `Schedule(config, init_ops=False)`，op 在调度过程中按需 `Operation(...)` + `register_operation()` 现场创建（因为一个 device 上同一 batch 会在两个 stage 出现，预建会冲突）。`waited_weight_grad` 队列实现 zero-bubble 的 W 延迟。

新增策略的最小改动路径：`src/strategies.py` 加一个 generator → `app.py` 的 `STRATEGIES` + `STRATEGY_INFO` 各加一项 → `main.py` 加一个 `run_*` 分支 → 若需要新的 stage 放置方式再动 `init_device_to_stages()`。

### 3. `src/visualizer.py` 与两个前端入口

- `convert_schedule_to_visualization_format(schedule)` 把 `Schedule` 转成 `{device_id: [task dict]}`，`create_pipeline_figure(data, max_time=...)` 生成 Plotly figure。**这两个是 app.py、main.py、examples/plot.py 三处共用的边界**，改签名要三处一起改。
- `visualize_pipeline_parallelism_dash()` 是 `main.py` 用的自带服务器版本（自动开浏览器，端口来自 `cfg.visualization_port`）。
- **`app.py`（约 1100 行）是独立的 Dash 应用**，不复用 `create_dash_app()`，只复用上面两个函数。它多策略并排对比、共享 `max_time` 轴、用 tabs 展示。部署入口是模块级的 `server = app.server`（gunicorn / HF Spaces，端口 7860），本地调试用 `run_app()`。
- 客户端交互（策略卡片选中、tab 切换）写在 `assets/clientside.js`，由 Dash 自动从 `assets/` 加载；样式在 `assets/custom.css`。

### `app.py` 里的隐式行为（改 UI 时务必注意）

`update_graph` 回调不是纯粹的参数透传，它会**自动修正用户输入**并弹 toast 提示：

- `1f1b` / `1f1b_overlap` / `zb1p` 强制 `num_stages = num_devices`；`dualpipe` 同理；`dualpipe_v` 强制 `num_stages = 2 * num_devices`。
- **op 时间会按 `1/stages_per_device` 缩放**——UI 里填的是"整个 device 一轮的耗时"，传给 `ScheduleConfig` 的是"单个 stage 的耗时"。这样不同 `num_stages` 的策略才能公平对比。新增策略时别忘了这个换算。
- `split_backward` 由策略名硬编码决定（`zb1p`/`dualpipe`/`dualpipe_v` 为 True），它同时决定校验哪些时间输入必填。

## 修改约定

- **改动尽量小、尽量局部**。这个仓库的核心价值在于调度语义的正确性，不在于代码整洁度；不要顺手重构 `get_dependencies()`、`execute()` 或已验证过的策略实现。发现无关的坏味道，先说出来，别自己动手。
- 时间语义只有一处来源：策略层排顺序，`execute()` 算时间。**任何在策略里直接写 start/end time 的做法都是错的。**
- 新增/修改策略后，用 `formula.py` 的方式做交叉验证（解析公式 vs 模拟结果），或至少跑一遍对应的 CLI 命令确认 assert 不炸、总时间合理。
- 改 `app.py` 的回调时，跟着补 `tests/test_app_dualpipe_v.py` 风格的测试：直接调用 `validate_inputs` / `update_graph` 函数并断言返回的组件树，不需要起浏览器。
- `.gitignore` 忽略 `*.json` 和 `uv.lock`；`assets/*.png` 走 git-lfs。Hydra 运行会往 `outputs/` 写工作目录（已忽略）。
