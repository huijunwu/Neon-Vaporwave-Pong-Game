# 如何 Host / Serve ONNX 版游戏

ONNX 版游戏需要通过 HTTP 服务器访问，不能直接用 `file://` 打开。
原因：`onnxruntime-web` 需要通过 `fetch` 加载 `.onnx` 模型和 `.wasm` 文件，浏览器安全策略禁止 `file://` 协议下的跨文件请求。

---

## 本地开发

首先安装依赖，然后启动 HTTP 服务器：

```bash
cd dist
npm install
```

### Python

```bash
python -m http.server 8080
```

### Node.js (npx)

```bash
npx serve dist -l 8080
```

### Node.js (http-server)

```bash
npx http-server dist -p 8080
```

启动后访问 http://localhost:8080

---

## 部署到公网

`dist/` 目录是完整的静态站点，可以直接部署到任意静态 hosting 服务：

### GitHub Pages

1. 在 repo Settings → Pages → Source 选择 `main` branch，目录选 `/dist`
2. 保存后访问 `https://<username>.github.io/<repo-name>/`

### Cloudflare Pages / Vercel / Netlify

1. 连接 GitHub repo
2. 设置 build output 目录为 `dist`（无需 build 命令）
3. 部署

---

## dist 目录结构

```
dist/
├── index.html                  入口页面
├── js/
│   ├── script.js               主线程游戏逻辑
│   └── onnx-worker.mjs         ONNX 推理 Worker（ES module）
├── css/
│   └── style.css               样式
├── package.json                npm 依赖（onnxruntime-web）
├── assets/
│   ├── images/                 图片资源
│   │   ├── codepong-title.png
│   │   └── codepong26.png
│   └── onnx/                   ONNX 模型（2 个）
│       ├── pong_step.onnx      物理引擎
│       └── pong_policy.onnx    AI 决策
└── node_modules/               npm install 后生成（.gitignore）
    └── onnxruntime-web/        WASM 运行时 + ONNX Runtime
```

`onnxruntime-web` 通过 npm 安装到本地 `node_modules/`，Worker 以 ES module 方式 import。

---

## 作为 Python 库使用（RL 训练）

`codepong26` 可以安装为 Python 包，其他 repo 可以依赖它做强化学习训练。

### 安装

```bash
pip install -e /path/to/Neon-Vaporwave-Pong-Game/src/python
```

### 快速开始：rule-based agent 对打

```python
import torch
from codepong26.step_module import PongStepModule
from codepong26.policy_module import PongPolicyModule

env = PongStepModule()
left = PongPolicyModule(reaction=0.16)
right = PongPolicyModule(reaction=0.20)

seed = torch.tensor(42, dtype=torch.int64)
state, ts = env.reset(seed)
ls = left.initial_state(torch.tensor(1, dtype=torch.int64))
rs = right.initial_state(torch.tensor(2, dtype=torch.int64))

for step in range(10000):
    la, ls = left.act(ts.obs[0], ls)
    ra, rs = right.act(ts.obs[1], rs)
    state, ts = env.step(state, torch.stack([la, ra]))

    if ts.done[0]:
        print(f"Game over at step {step}: {state.score_left.item():.0f}-{state.score_right.item():.0f}")
        state, ts = env.reset(torch.tensor(step, dtype=torch.int64))
        ls = left.initial_state(torch.tensor(step + 1, dtype=torch.int64))
        rs = right.initial_state(torch.tensor(step + 2, dtype=torch.int64))
```

### 训练一个简单的 RL agent

```python
import torch
import torch.nn as nn
import torch.optim as optim
from codepong26.step_module import PongStepModule
from codepong26.policy_module import PongPolicyModule
from codepong26.functional import Timestep

# 神经网络 policy（替换 rule-based）
class NNPolicy(nn.Module):
    def __init__(self, obs_dim=6, n_actions=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, obs):
        logits = self.net(obs)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)

# 训练循环
env = PongStepModule()
opponent = PongPolicyModule(reaction=0.16)  # rule-based 对手
learner = NNPolicy()
optimizer = optim.Adam(learner.parameters(), lr=3e-4)

seed = torch.tensor(0, dtype=torch.int64)
state, ts = env.reset(seed)
opp_state = opponent.initial_state(torch.tensor(1, dtype=torch.int64))

rewards_log = []
log_probs_log = []

for step in range(50000):
    # learner 控制左边球拍
    action_l, log_prob = learner(ts.obs[0])

    # opponent 控制右边球拍
    action_r, opp_state = opponent.act(ts.obs[1], opp_state)

    state, ts = env.step(state, torch.stack([action_l.float(), action_r]))

    rewards_log.append(ts.reward[0])  # learner 的 reward
    log_probs_log.append(log_prob)

    # 每局结束更新一次（REINFORCE）
    if ts.done[0]:
        returns = torch.stack(rewards_log)
        log_probs = torch.stack(log_probs_log)

        # 简单的 REINFORCE
        loss = -(log_probs * returns).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 < 100:
            print(f"Step {step}, Score: {state.score_left.item():.0f}-{state.score_right.item():.0f}")

        # 重置
        rewards_log.clear()
        log_probs_log.clear()
        seed = torch.tensor(step, dtype=torch.int64)
        state, ts = env.reset(seed)
        opp_state = opponent.initial_state(torch.tensor(step + 1, dtype=torch.int64))
```

### 训练完成后导出为 ONNX

```python
# 将训练好的 NN policy 包装成 ONNX 可导出的 Module
class TrainedPolicy(nn.Module):
    def __init__(self, nn_policy):
        super().__init__()
        self.net = nn_policy.net

    def forward(self, obs, memory_y, rand_val, H):
        # memory_y 和 rand_val 不用（NN 是无状态的），保持接口兼容
        logits = self.net(obs)
        action = logits.argmax().float()
        return action, memory_y  # 返回格式和 PongPolicyModule 一致

# 导出
trained = TrainedPolicy(learner)
torch.onnx.export(
    trained,
    (torch.randn(6), torch.tensor(300.0), torch.tensor(0.0), torch.tensor(600.0)),
    "pong_policy.onnx",
    input_names=["obs", "memory_y", "rand_val", "H"],
    output_names=["action", "new_memory_y"],
    opset_version=17,
)
# 替换 dist/assets/onnx/pong_policy.onnx 即可在浏览器中运行训练好的 agent
```

### 支持 `torch.vmap`（批量训练加速）

```python
env = PongStepModule()
batch_size = 64

batched_reset = torch.vmap(lambda s: env.reset(s))
batched_step = torch.vmap(lambda s, a: env.step(s, a))

seeds = torch.arange(batch_size, dtype=torch.int64)
states, ts = batched_reset(seeds)           # 同时运行 64 个游戏

actions = torch.randint(0, 3, (batch_size, 2))
states, ts = batched_step(states, actions)  # 64 个游戏同时 step
```
