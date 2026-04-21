# RL 环境接口设计 — PyTorch Functional API

适用于 `torch.compile` + `torch.vmap` 的 end-to-end GPU 训练。

---

## 1. 设计动机

### 问题

传统 RL 环境接口（Gymnasium、PettingZoo）是有状态的面向对象 API：

```python
env.reset()       # 修改 self 内部状态
env.step(action)  # 修改 self 内部状态
```

这与 PyTorch 的两个加速原语根本冲突：

| 原语            | 要求                                    | 传统接口                          |
| --------------- | --------------------------------------- | --------------------------------- |
| `torch.compile` | 纯 tensor 计算图，无 Python side effect | `self.state = ...` 是 side effect |
| `torch.vmap`    | 函数对 batch 维度做向量化映射           | 有状态对象无法 vmap               |

### 目标

设计一个 **纯函数式** 环境接口：

- 所有方法都是纯函数：`(state, action) → (new_state, timestep)`
- State 和 Timestep 是 `NamedTuple of Tensors`，固定 shape
- 兼容 single-agent 和 multi-agent（MARL）
- 训练完的 policy network 导出 ONNX，浏览器推理

### 参考

| 项目     | 生态 | 借鉴点                                               |
| -------- | ---- | ---------------------------------------------------- |
| Gymnax   | JAX  | 函数式 single-agent 接口：`step(key, state, action)` |
| JaxMARL  | JAX  | MARL 扩展，actions/obs 用 dict per agent             |
| Parallax | JAX  | 最新最简 Protocol，agents 作为 tensor 维度           |

本接口是 **Parallax 的设计思路 + PyTorch 实现**。

---

## 2. 核心类型

### 2.1 Timestep

每次 `reset()` / `step()` 返回的观测数据包：

```python
class Timestep(NamedTuple):
    obs: Tensor          # (n_agents, *obs_shape)
    reward: Tensor       # (n_agents,)
    done: Tensor         # (n_agents,)  — bool
    truncated: Tensor    # (n_agents,)  — bool
    info: Tensor         # (n_agents, *info_shape) 或 scalar 0
```

所有字段都是 Tensor，第一个维度始终是 agent 维度。Single-agent 环境中 `n_agents=1`。

### 2.2 Space

轻量的 action/observation 空间描述，只携带 shape 信息：

```python
class Discrete(NamedTuple):
    n: int                    # 离散动作数量

class Box(NamedTuple):
    low: Tensor               # 下界
    high: Tensor              # 上界
    shape: tuple[int, ...]    # tensor shape

Space = Discrete | Box
```

不提供 `sample()` / `contains()` 方法 — 这些是 Gymnasium 的便利函数，对 compile 无意义。

### 2.3 EnvState

每个环境自定义的 `NamedTuple`，存储全部内部状态。例如 Pong：

```python
class PongState(NamedTuple):
    ball_x: Tensor
    ball_y: Tensor
    ball_vx: Tensor
    ball_vy: Tensor
    paddle_left_y: Tensor
    paddle_right_y: Tensor
    score_left: Tensor
    score_right: Tensor
    rally: Tensor
    step_count: Tensor
    seed: Tensor
```

要求：

- 每个字段都是 Tensor
- Shape 在整个 episode 内固定不变
- 用 `NamedTuple`（不是 `dataclass`），因为 PyTorch pytree 原生支持 NamedTuple 的 flatten/unflatten，`vmap` 可以自动沿 batch 维度展开和重组

---

## 3. Env Protocol

```python
@runtime_checkable
class Env(Protocol):

    @property
    def n_agents(self) -> int: ...

    @property
    def obs_space(self) -> Space: ...

    @property
    def action_space(self) -> Space: ...

    def reset(self, seed: Tensor) -> tuple[Any, Timestep]: ...

    def step(self, state: Any, actions: Tensor) -> tuple[Any, Timestep]: ...

    def reset_done(self, state: Any, timestep: Timestep, seed: Tensor) -> tuple[Any, Timestep]: ...
```

### 3.1 `reset(seed) → (state, timestep)`

初始化环境，返回初始 state 和 observation。

- `seed` 是 `int64` scalar tensor（不是 `torch.Generator`，后者无法 vmap）
- 返回的 `state` 是环境自定义的 NamedTuple
- 每次调用结果完全由 seed 决定（deterministic）

### 3.2 `step(state, actions) → (new_state, timestep)`

核心：纯函数式状态转移，**无任何 side effect**。

- `state`：上一次 `reset()` 或 `step()` 返回的 state
- `actions`：`(n_agents,)` for Discrete / `(n_agents, *act_shape)` for Box
- 返回全新的 `(new_state, timestep)`，原 state 不被修改

### 3.3 `reset_done(state, timestep, seed) → (state, timestep)`

在 vmap batch 中自动 reset 已结束的环境。

默认实现（`auto_reset`）：当任一 agent 的 `done=True`，用 `torch.where` 选择 fresh state 替换 current state。无 Python `if/else` 分支。

---

## 4. compile + vmap 兼容性约束

实现环境时必须遵守以下规则，否则 `torch.compile(fullgraph=True)` 或 `torch.vmap` 会失败：

### 4.1 禁止依赖 tensor 值的 Python 控制流

```python
# 错误 — Python if 依赖 tensor 值，compile 无法追踪
if ball_y < 0:
    ball_y = 0
    ball_vy = -ball_vy

# 正确 — torch.where 是纯 tensor 运算
hit = ball_y < 0
ball_y = torch.where(hit, torch.tensor(0.0), ball_y)
ball_vy = torch.where(hit, -ball_vy, ball_vy)
```

### 4.2 禁止 side effect

```python
# 错误 — 修改 self 属性
self.state = new_state

# 正确 — 返回新值
return new_state, timestep
```

### 4.3 固定 shape

所有 Tensor 的 shape 在整个 episode 内不能变化。不能动态增删 tensor 维度。

### 4.4 RNG 处理

`torch.Generator` 不支持 vmap。使用显式 seed tensor + 纯 tensor 数学运算生成伪随机数：

```python
def split_seed(seed: Tensor, n: int) -> tuple[Tensor, ...]:
    """用 Knuth 乘法哈希从一个 seed 派生 n 个子 seed"""
    return tuple((seed * 2654435761 + i) % (2**31) for i in range(n))

def manual_uniform(seed: Tensor) -> Tensor:
    """LCG 伪随机数，纯 tensor 运算，vmap-safe"""
    ...
```

---

## 5. Shape 约定

### 单环境（vmap 之前）

```
obs:       (n_agents, *obs_shape)    例: Pong → (2, 6)
reward:    (n_agents,)               例: Pong → (2,)
done:      (n_agents,)               例: Pong → (2,)
actions:   (n_agents,)               例: Pong → (2,)
```

### vmap batch 后

`torch.vmap` 在最外层自动添加 batch 维度：

```
obs:       (B, n_agents, *obs_shape)  例: B=16, Pong → (16, 2, 6)
reward:    (B, n_agents,)             例: (16, 2)
actions:   (B, n_agents,)             例: (16, 2)
```

### Single-agent 环境

`n_agents=1`，agent 维度仍然存在：

```
obs:       (1, *obs_shape)    例: Flappy Bird → (1, 4)
actions:   (1,)               例: (1,)
```

---

## 6. 使用方式

### 6.1 基础 rollout

```python
env = PongEnv()
seed = torch.tensor(42, dtype=torch.int64)
state, ts = env.reset(seed)

for _ in range(1000):
    actions = policy(ts.obs)           # (2,)
    state, ts = env.step(state, actions)
    if ts.done.any():
        state, ts = env.reset_done(state, ts, seed)
```

### 6.2 vmap 并行多环境

```python
env = PongEnv()
batched_step = torch.vmap(lambda s, a: env.step(s, a))
batched_reset = torch.vmap(lambda s: env.reset(s))

seeds = torch.arange(64, dtype=torch.int64)
states, ts = batched_reset(seeds)             # (64, 2, 6)

actions = torch.randint(0, 3, (64, 2))
states, ts = batched_step(states, actions)    # 64 个环境并行 step
```

### 6.3 compile + vmap

```python
compiled_step = torch.compile(
    torch.vmap(lambda s, a: env.step(s, a)),
    fullgraph=True,
    backend="aot_eager",  # 或 "inductor"（需要 C++ 编译器）
)

for _ in range(1000):
    actions = policy(ts.obs)
    states, ts = compiled_step(states, actions)
```

---

## 7. Agent Protocol

### 接口定义

```python
@runtime_checkable
class Agent(Protocol):
    def act(self, obs: Tensor, state: Any | None = None) -> tuple[Tensor, Any | None]:
        ...
```

- `obs`：单个 agent 的观测，shape `(*obs_shape)`（无 agent dim，无 batch dim）
- `state`：agent 内部状态（rule-based 的 memory、RNN hidden 等）。首次调用传 `None`。
- 返回 `(action, new_state)`：action 是 scalar int64 tensor（Discrete 空间）

### 设计意图

Agent 和 Env 是**正交的**：

```
Env：物理世界，接受 action，返回 obs/reward/done
Agent：决策者，接受 obs，返回 action
```

替换 agent 不需要改 env，替换 env 不需要改 agent。将来用神经网络替换 rule-based AI 只需要：

```python
# 现在
left_agent = RuleBasedAgent(reaction=0.16)

# 将来（只改这一行）
left_agent = NNAgent(policy_net)
```

### RuleBasedAgent

包装 `models.py` 的 `AIUpdate` 逻辑，输出离散 action：

1. 从归一化 obs 恢复物理坐标（ball_y、ball_vy、own_paddle_y）
2. 用 `AIUpdate` 的 look-ahead + jitter + smooth tracking 计算 target_y
3. 比较 `target_y` 和 `paddle_y` 的差值，转换为离散 action：
   - `diff > threshold` → DOWN (2)
   - `diff < -threshold` → UP (1)
   - else → NOOP (0)

内部状态 `RuleBasedState` 包含 `memory_y`（tracking 记忆）和 `seed`（jitter 随机数）。

---

## 8. 物理逻辑复用（Single Source of Truth）

### 问题

重构前有两份独立的 Pong 物理代码：

- `models.py` 的 `BallPhysics`（nn.Module，ONNX 导出给浏览器）
- `rl/envs/pong.py` 的 `PongEnv.step()`（RL 训练用）

两份代码可能不一致，修一个忘了修另一个。

### 解决方案

提取纯函数到 `rl/physics.py` 作为 single source of truth：

```
rl/physics.py（纯函数）
  ├── ball_move()
  ├── wall_collide()
  ├── paddle_collide()
  ├── score_detect()
  └── apply_action()
        │
        ├── rl/envs/pong.py PongEnv.step() 调用这些函数
        └── models.py BallPhysics.forward() 可以迁移为调用这些函数
```

`models.py` 当前仍保持独立（为了不破坏已有的 ONNX 导出），但将来可以迁移为调用 `rl/physics.py`。

---

## 9. 文件结构

```
src/python/rl/
├── __init__.py              公共导出
├── agent_protocol.py        Agent Protocol 定义
├── env_protocol.py          Env Protocol 定义
├── timestep.py              Timestep NamedTuple
├── spaces.py                Discrete / Box
├── physics.py               Pong 物理纯函数（single source of truth）
├── functional.py            vmap-safe RNG + auto_reset
├── agents/
│   ├── __init__.py
│   └── rule_based.py        RuleBasedAgent（包装 AIUpdate 为离散 action）
├── envs/
│   ├── __init__.py
│   └── pong.py              PongEnv（2-agent competitive）
└── test_smoke.py            5 项验证测试
```

---

## 10. 设计决策记录

| 决策            | 选择                                    | 备选项                               | 理由                                                                                                                 |
| --------------- | --------------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| State 类型      | NamedTuple                              | dataclass / flat tensor / TensorDict | pytree 原生支持，vmap 自动展开重组；dataclass 需要额外注册；flat tensor 失去字段名可读性；TensorDict 与 compile 冲突 |
| SARL/MARL 统一  | 统一接口，`n_agents>=1`                 | 分开两套接口                         | 避免维护两套 API；single-agent 就是 agent 维度 = 1                                                                   |
| RNG             | int64 seed + LCG                        | torch.Generator                      | Generator 无法 vmap；显式 seed 是纯 tensor 运算                                                                      |
| Agent 维度      | dim=0 within per-agent tensors          | dict per agent (JaxMARL 风格)        | tensor 维度更 vmap 友好；dict 需要对齐 key                                                                           |
| Space 描述      | 轻量 NamedTuple                         | Gymnasium Space                      | 不需要 sample/contains；减少依赖                                                                                     |
| Auto-reset      | `torch.where` 选择 fresh/current        | Python if per env                    | vmap batch 内每个 env 独立 reset，无分支                                                                             |
| compile backend | `aot_eager`（开发）/ `inductor`（生产） | —                                    | aot_eager 验证 fullgraph 追踪能力，不依赖 C++ 编译器                                                                 |

---

## 11. 与现有方案对比

|           | PettingZoo | TorchRL                    | Gymnax/JaxMARL | 本方案                       |
| --------- | ---------- | -------------------------- | -------------- | ---------------------------- |
| 语言      | Python     | Python+TensorDict          | JAX            | PyTorch                      |
| 状态管理  | 有状态 OOP | 有状态 OOP                 | 纯函数         | 纯函数                       |
| compile   | 不兼容     | 部分兼容（已知 bug）       | N/A (JAX jit)  | 完全兼容                     |
| vmap      | 不兼容     | 多进程并行                 | vmap 兼容      | vmap 兼容                    |
| ONNX 导出 | N/A        | N/A                        | JAX→ONNX 困难  | `torch.onnx.export` 一等公民 |
| 复杂度    | 中         | 高（TensorDict/spec 系统） | 低             | 低                           |
| MARL      | 标准接口   | 支持                       | 支持           | 支持                         |

---

## 12. 训练与部署的边界

本接口服务于两个 goal：

1. **训练阶段**：env + agent 的 policy network 全程 GPU，不回 CPU
2. **部署阶段**：env 物理逻辑 + policy network **都**导出 ONNX，放入浏览器推理

### 整体架构

```
┌──────────────────────────────────────────────────────────┐
│  训练侧（Python + PyTorch, GPU）                         │
│                                                          │
│   PongEnv.step(state, actions)   ← 环境物理，纯 tensor   │
│        ↕                                                 │
│   PolicyNetwork(obs) → actions   ← nn.Module             │
│        ↕                                                 │
│   PPO / DQN loss + optimizer     ← 标准 PyTorch          │
│                                                          │
│   全部通过 compile + vmap 加速，不离开 GPU               │
└──────────────────────────────────────────────────────────┘
          │                              │
          │ torch.onnx.export            │ torch.onnx.export
          │ (env step 拆分为模块)        │ (policy network)
          ▼                              ▼
┌──────────────────────────────────────────────────────────┐
│  部署侧（浏览器, onnx-runtime-web）                      │
│                                                          │
│   pong_physics.onnx   ← 球运动 + 碰撞 + 得分             │
│   pong_policy.onnx    ← agent 决策                       │
│                                                          │
│   JS 渲染层只负责画面和输入，所有计算在 ONNX 中完成      │
└──────────────────────────────────────────────────────────┘
```

本项目已有先例：`models.py` 中的 `BallPhysics`、`AIUpdate` 等就是把游戏物理逻辑导出为 ONNX 在浏览器运行（见 `export_onnx.py`）。RL 训练后的部署遵循同样模式。

### 环境导出为什么可行

环境接口的设计约束（纯 tensor 运算、`torch.where` 替代 if/else、固定 shape）恰好也是 `torch.onnx.export` 的要求。也就是说：

> **compile-friendly ≈ ONNX-exportable**
>
> 满足 `torch.compile(fullgraph=True)` 的代码，基本上也满足 ONNX 导出的要求。

### 导出策略

环境的 `step()` 是一个大函数（物理 + 碰撞 + 得分 + re-serve），导出时有两种策略：

| 策略     | 做法                                                 | 优点                              | 缺点                            |
| -------- | ---------------------------------------------------- | --------------------------------- | ------------------------------- |
| 整体导出 | `step()` 整个导出为一个 ONNX                         | 简单                              | ONNX 模型较大，浏览器端调试困难 |
| 拆分导出 | 按职责拆分：physics.onnx + scoring.onnx + serve.onnx | 模块化，与现有 models.py 模式一致 | 需要在 JS 端编排多个 ONNX 调用  |

**推荐拆分导出**，与 `models.py` 的现有模式保持一致。训练时用完整的 `PongEnv.step()` 做 compile + vmap 加速，导出时拆分成浏览器端需要的粒度。

### 导出示例

**Policy network 导出**（标准 nn.Module，直接导出）：

```python
policy_net = PolicyNetwork(obs_dim=6, n_actions=3)

dummy_obs = torch.randn(1, 6)
torch.onnx.export(
    policy_net, dummy_obs,
    "pong_policy.onnx",
    input_names=["obs"],
    output_names=["action_logits"],
    opset_version=17,
)
```

**环境物理导出**（包一层 nn.Module，与 models.py 现有模式一致）：

```python
class PongStepModule(nn.Module):
    """将 PongEnv.step() 的物理部分包装为 nn.Module 以便 ONNX 导出。"""
    def forward(self, ball_x, ball_y, ball_vx, ball_vy,
                paddle_left_y, paddle_right_y,
                action_left, action_right):
        # ... 复用 PongEnv.step() 中的 tensor 运算 ...
        return new_ball_x, new_ball_y, new_ball_vx, new_ball_vy, ...
```

**浏览器端调用**：

```javascript
// 每帧：env step → 观测 → policy → 动作 → 下一帧
const envResult = await envSession.run({ ball_x, ball_y, ... });
const obs = buildObs(envResult);
const policyResult = await policySession.run({ obs });
const action = argmax(policyResult.action_logits.data);
```

### 训练 vs 导出的环境区别

| 关注点             | 训练时（PongEnv）          | 导出时（ONNX Module）             |
| ------------------ | -------------------------- | --------------------------------- |
| RNG（re-serve）    | 需要，用 seed tensor + LCG | 不需要，浏览器端用 JS Math.random |
| Reward 计算        | 需要                       | 不需要，浏览器不做训练            |
| done / truncated   | 需要                       | 不需要                            |
| Observation 归一化 | 需要（agent 输入）         | 需要（同样的归一化）              |
| 物理 + 碰撞        | 需要                       | 需要（核心导出内容）              |

训练环境是导出环境的**超集**。导出时去掉 RL-only 的部分（reward、done、auto-reset），保留物理核心。

---

## 13. GPU end-to-end 与 zero CPU↔GPU sync

### "不离开 GPU" 到底意味着什么

"end-to-end GPU 训练" 的精确定义是：**整个训练 loop（env step → policy forward → loss → backward → optimizer）中没有任何 CPU↔GPU sync point**。

`torch.compile + torch.vmap` 是主要手段，但**不是充要条件**：

```
"end-to-end 不离开 GPU，无 CPU↔GPU sync"
    ≠ torch.compile + torch.vmap（只是手段）
    = 以下全部满足：

    1. env.step()      纯 tensor 运算        ← compile + vmap 保证
    2. policy forward   纯 tensor 运算        ← 标准 nn.Module
    3. reset_done       无条件每步调用        ← torch.where，不是 Python if
    4. 训练 loop        无 .item() / .cpu()   ← 写法约束
    5. loss + backward  不间断 GPU            ← 标准 PyTorch
```

### 隐蔽的 sync point

即使 env 和 policy 都通过了 compile + vmap 验证，以下写法仍会触发 CPU↔GPU sync：

| 写法                                            | 为什么触发 sync                                   | 怎么改                                             |
| ----------------------------------------------- | ------------------------------------------------- | -------------------------------------------------- |
| `if ts.done.any(): reset()`                     | `.any()` 把 GPU tensor 拉回 CPU 给 Python if 判断 | 无条件调用 `reset_done()`，内部 `torch.where` 处理 |
| `loss.item()` 用于 logging                      | `.item()` 触发 sync                               | 用 `loss.detach()` 累积，每 N 步才 `.item()` 一次  |
| `print(reward)`                                 | 同上                                              | 去掉，或低频 log                                   |
| `if step > warmup:` 当 step 是 tensor           | tensor → Python bool 触发 sync                    | step 用 Python int，不用 tensor                    |
| `buffer[idx] = transition` 当 idx 是 GPU tensor | GPU tensor 做 Python indexing                     | buffer 全程用 `torch.scatter` 或预分配固定 shape   |

### 正确的 zero-sync 训练 loop

```python
env = PongEnv()
policy = PolicyNetwork(...)
optimizer = torch.optim.Adam(policy.parameters())

batched_step = torch.vmap(lambda s, a: env.step(s, a))
batched_reset_done = torch.vmap(lambda s, ts, seed: env.reset_done(s, ts, seed))

compiled_step = torch.compile(batched_step, fullgraph=True)
compiled_reset = torch.compile(batched_reset_done, fullgraph=True)

seeds = torch.arange(batch_size, dtype=torch.int64, device="cuda")
states, ts = torch.vmap(lambda s: env.reset(s))(seeds)

# 预分配 rollout buffer — 固定 shape，全程 GPU
obs_buf   = torch.zeros(rollout_len, batch_size, 2, 6, device="cuda")
act_buf   = torch.zeros(rollout_len, batch_size, 2, dtype=torch.int64, device="cuda")
rew_buf   = torch.zeros(rollout_len, batch_size, 2, device="cuda")
done_buf  = torch.zeros(rollout_len, batch_size, 2, dtype=torch.bool, device="cuda")

for epoch in range(n_epochs):
    # ── Rollout（zero sync）──
    for t in range(rollout_len):           # Python int loop，不触发 sync
        actions = policy(ts.obs)           # GPU → GPU
        states, ts = compiled_step(states, actions)

        obs_buf[t] = ts.obs                # GPU tensor 赋值，无 sync
        act_buf[t] = actions
        rew_buf[t] = ts.reward
        done_buf[t] = ts.done

        # 无条件 reset — 不需要 if ts.done.any()
        states, ts = compiled_reset(states, ts, seeds)

    # ── 训练（zero sync）──
    loss = ppo_loss(obs_buf, act_buf, rew_buf, done_buf, policy)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # ── Logging（低频 sync，不影响训练性能）──
    if epoch % 100 == 0:
        print(f"epoch {epoch}, loss: {loss.item():.4f}")  # 仅此处 sync
```

关键区别：

| 写法           | 有 sync                             | zero sync                       |
| -------------- | ----------------------------------- | ------------------------------- |
| done 检查      | `if ts.done.any(): reset()`         | 无条件 `reset_done()`，每步都调 |
| loss logging   | 每步 `loss.item()`                  | 每 N 步才 `.item()`             |
| rollout buffer | Python list append                  | 预分配固定 shape tensor         |
| for 循环       | `for t in range(N)` (Python int) ✅ | 同左 ✅                         |

### 接口设计如何支撑 zero sync

本接口的三个设计决策直接服务于 zero sync：

1. **`reset_done()` 用 `torch.where`**：可以无条件每步调用，done 的环境被 reset，没 done 的不变。不需要 Python if 判断 done 状态。

2. **纯函数式 step**：`(state, actions) → (new_state, timestep)`，无 side effect。可以被 compile 编译为单个 fused kernel，中间无 Python 回调。

3. **NamedTuple state**：pytree 兼容，vmap 自动处理 batch 维度。不需要手动拆包/拼包触发 sync。

### compile + vmap 在这个体系中的角色

```
torch.compile   →  消除 Python 解释器 overhead，fuse env.step() 为 GPU kernel
torch.vmap      →  把 for loop over envs 变成 batch tensor 运算
zero-sync 写法  →  消除训练 loop 中的 CPU↔GPU 数据搬运

三者缺一不可。compile + vmap 保证"单步不离开 GPU"，
zero-sync 写法保证"loop 级别不离开 GPU"。
```

### 接口的 device 策略

接口本身是 **device-agnostic** 的 — 所有运算都是纯 tensor 操作（`torch.where`、`torch.clamp`、加减乘除），天然支持任意 device。

在 GPU 上运行只需要确保所有 tensor 创建在同一 device 上：

```python
device = torch.device("cuda")

seed = torch.tensor(42, dtype=torch.int64, device=device)
state, ts = env.reset(seed)

actions = policy(ts.obs)
state, ts = env.step(state, actions)  # 全程 GPU，无 CPU↔GPU transfer
```

### 当前实现的限制

环境代码中有 `torch.tensor(BALL_R)` 这类字面量创建，会默认在 CPU 上。要实现真正的 GPU end-to-end，需要：

1. 环境的常量 tensor 在初始化时 `.to(device)`，或在 `reset()`/`step()` 中从输入 tensor 推断 device
2. 用 `torch.compile(backend="inductor")` 而非 `aot_eager`（后者不做 kernel fusion）

这属于实现优化，不改变接口设计。接口层面已经满足 GPU end-to-end 的所有约束。

### compile backend 选择

| backend     | 用途      | 特点                                                          |
| ----------- | --------- | ------------------------------------------------------------- |
| `aot_eager` | 开发/测试 | 验证 fullgraph 可追踪性，不需要 C++ 编译器，不做优化          |
| `inductor`  | 生产训练  | kernel fusion + CUDA codegen，真正的性能提升，需要 C++ 编译器 |

当前测试用 `aot_eager` 证明了计算图可以被完整追踪（`fullgraph=True`），这是最关键的验证 — 说明代码中没有 compile-breaking 的 Python 控制流。切换到 `inductor` 只是环境配置问题。

---

## 14. 验证结果

使用 `rl/test_smoke.py` 运行 5 项测试（CPU，`aot_eager` backend）：

| 测试            | 内容                                            | 结果                    |
| --------------- | ----------------------------------------------- | ----------------------- |
| basic rollout   | 200 步手动 rollout                              | 通过                    |
| multi-agent     | 2 个 RuleBasedAgent 对打 5000 步                | 通过，产生真实比分      |
| vmap batch      | 8 个环境并行 reset + step                       | 通过，shape `(8, 2, 6)` |
| compile         | `fullgraph=True` 完整计算图追踪                 | 通过                    |
| compile + vmap  | 16 个环境 × 100 步，编译后批量执行              | 通过，正常产生得分      |

```bash
cd src/python && PYTHONPATH=. python rl/test_smoke.py
```

> **注**：测试在 CPU + `aot_eager` 上完成。`aot_eager` 验证的是 fullgraph 追踪能力（无 graph break），这是 GPU end-to-end 的前提条件。CUDA + `inductor` 的完整 GPU 验证待有 CUDA 环境时补充。
