# Code Flow — ONNX 版本 (dist)

JS + ONNX Runtime Web 实现，双 AI 互搏（watch mode）。
物理引擎和 AI 决策由 2 个 ONNX 模型执行，渲染/音频/特效留在 JS。

---

## 1. 启动流程

```
浏览器加载 index.html
  └─ <script src="js/script.js">
       ├─ 创建 Worker("js/onnx-worker.mjs", {type:"module"})
       │   └─ Worker 内部: import ort from node_modules (ES module)
       │   └─ import.meta.url 自动解析 WASM 路径
       ├─ 注册 DOM 引用（canvas, overlay, buttons...）
       ├─ 定义 ONNX wrapper 函数（onnxPolicy, onnxStep）
       ├─ 定义 JS 原生函数（clamp, lerp, shock, updateCutscene, updateParticles）
       ├─ 注册事件监听器（click, keydown — 无鼠标/球拍控制）
       └─ init()                                        async 入口
            ├─ resize() / 初始化状态
            ├─ overlayImg.src = INTRO_IMG
            ├─ playText = "LOADING…"                    ← 加载提示
            ├─ await workerInit()                       Worker 加载 2 个 ONNX 模型
            │   ├─ pong_step.onnx      (10.8K) — 物理引擎
            │   └─ pong_policy.onnx    (2.7K)  — AI 决策
            ├─ playText = "START"                       ← 加载完成
            └─ requestAnimationFrame(update)            游戏循环开始
```

**耗时**: 数百毫秒（npm 依赖 + 2 个 .onnx + WASM runtime），用户先看到 "LOADING…" 再变 "START"。

---

## 2. 游戏循环 (`update` — 每帧 ~16ms)

```
async update(ts)                                     ← requestAnimationFrame 回调
│
├─ requestAnimationFrame(update)                     排队下一帧
│
├─ dt 计算
│   dt = clamp(raw_dt, 0.008, 0.02)                  JS 原生函数
│   dt *= state.slowmo
│
├─ 全局状态更新
│   state.time += dt
│   state.shake = Math.max(0, shake - dt * 22)       JS 内联
│
├─ updateCutscene(dt)                                JS 原生函数
│
├─ if (!running || paused) → draw() → return
│
├─ 双 AI 决策 ← ONNX
│   ├─ obsL = buildObs(true)                         JS 构建归一化观测向量
│   ├─ obsR = buildObs(false)                        JS 构建归一化观测向量
│   ├─ leftResult = await onnxPolicy(obsL, ...)      ☆ ONNX: pong_policy.onnx
│   │   └─ 输入: obs[6], memory_y, rand_val, H
│   │   └─ 输出: action (0/1/2), new_memory_y
│   └─ rightResult = await onnxPolicy(obsR, ...)     ☆ ONNX: pong_policy.onnx
│       └─ 同上（同一模型，不同输入）
│
├─ 物理步进 ← ONNX
│   └─ stepResult = await onnxStep(...)              ☆ ONNX: pong_step.onnx
│       ├─ 输入: ball 状态, 两个球拍 Y, 两个 action, rally, W, H
│       ├─ 内部执行:
│       │   ├─ apply_action() × 2 — 球拍移动
│       │   ├─ ball_move() — 球位移
│       │   ├─ wall_collide() — 上下墙壁碰撞
│       │   ├─ paddle_collide() × 2 — 左右球拍碰撞 + 速度递增
│       │   └─ score_detect() — 出界判定
│       └─ 输出: 新状态 + events[6]
│
├─ 拖尾
│   trail.push({ x, y, t })                          JS 数组 push
│
├─ 处理 events（来自 ONNX 的 float32[6] 标记）
│   ├─ wallTop/wallBottom:
│   │   sparkLine(); beep("wall")                    JS（rand + Canvas + Audio）
│   ├─ hitLeft/hitRight:
│   │   burst(); sparkLine(); shock()                JS（rand + Canvas）
│   │   beep("hit")                                  JS（Audio）
│   └─ scoredL/scoredR:
│       handleScore(side, dir)                       JS
│         score++; rally = 0; burst(); shock()
│         beep("score")
│         checkWinOrReset() → resetRound() 或 endGame()
│
├─ updateParticles(dt)                               JS 原生函数
│
└─ draw()                                            JS 函数调用
    ├─ clearRect
    ├─ 屏幕震动 translate
    ├─ glow 背景（3 个径向渐变）                     Canvas API
    ├─ 中线虚线                                      Canvas API
    ├─ 拖尾渲染                                      Canvas API
    ├─ drawPaddle() × 2                              Canvas API
    ├─ drawBall()                                    Canvas API
    └─ drawParticles()                               Canvas API
        ├─ for (i < 70) if alive: fillRect           遍历 Float32Array
        └─ for (i < 40) if alive: moveTo/lineTo      遍历 Float32Array
```

---

## 3. 用户交互

```
Watch mode — 无鼠标/球拍控制，仅键盘快捷键

键盘按下
  ├─ Space → start() 或 togglePause()                async
  ├─ R → hardRestart() (= start 的别名)              async
  └─ M → toggleSound()                               同步

按钮点击
  ├─ START → start()                                 async
  │   └─ resetGame() → resetRound()
  ├─ PAUSE → togglePause()                           同步
  ├─ SOUND → toggleSound()                           同步
  └─ FX → toggleFX()                                 同步
       └─ particleBuf.fill(0); sparkBuf.fill(0)      清空固定缓冲区
```

---

## 4. 数据流

```
输入层              计算层                                输出层
─────────          ─────────                             ─────────
ball state ───────→ buildObs() → 归一化 obs[6] ────────→ (JS 变量)
obs + memoryY ────→ ☆ ONNX policy → action + memoryY ──→ (JS 变量)
obs + memoryY ────→ ☆ ONNX policy → action + memoryY ──→ (JS 变量)
state + actions ──→ ☆ ONNX step → 新 state + events ───→ ball/paddle → draw()
events[6] ────────→ burst() / sparkLine() ─────────────→ particleBuf → drawParticles()
events[6] ────────→ beep() ────────────────────────────→ Web Audio
events[6] ────────→ shock() ───────────────────────────→ state.shake
events[6] ────────→ handleScore() → score++ ───────────→ DOM 更新
particleBuf ──────→ updateParticles() (JS) ────────────→ particleBuf → drawParticles()
sparkBuf ─────────→ updateParticles() (JS) ────────────→ sparkBuf → drawParticles()
```

---

## 5. 关键特征

| 特征        | 说明                                                             |
| ----------- | ---------------------------------------------------------------- |
| 执行模型    | **async/await**，ONNX 推理在 Web Worker 线程（WASM 后端）        |
| 线程模型    | 主线程（渲染+输入）+ Worker 线程（ONNX 推理），postMessage 通信  |
| ONNX 模型   | 2 个：`pong_step`（物理）+ `pong_policy`（AI），每帧 3 次调用    |
| JS 原生函数 | clamp, lerp, shock, updateCutscene, updateParticles（不走 ONNX） |
| 游戏模式    | Watch mode — 双 AI 互搏，无玩家操控                              |
| 动态尺寸    | W/H 作为模型参数传入，适配任意 Canvas 尺寸                       |
| 球速        | 每次击球 +15，无上限，得分后 reset 到 560                        |
| 启动时间    | 数百毫秒（需 `npm install` + 加载 2 个 .onnx + WASM runtime）    |
| 粒子存储    | `Float32Array` 固定缓冲区 [70×7] / [40×7]，alive 掩码管理        |
| 分支逻辑    | ONNX 内部用 `Where` 算子（等价 `torch.where`）                   |
| 外部依赖    | Google Fonts, Font Awesome, 图片（本地），onnxruntime-web（npm） |
| 运行方式    | `npm install` + HTTP 服务器，不支持 `file://`                    |

---

## 6. 目录结构

```
dist/
├── index.html                  入口页面
├── js/
│   ├── script.js               主线程游戏逻辑
│   └── onnx-worker.mjs         ONNX 推理 Worker（ES module）
├── css/
│   └── style.css               样式
├── assets/
│   ├── images/                 图片资源
│   │   ├── codepong-title.png
│   │   └── codepong26.png
│   └── onnx/                   ONNX 模型
│       ├── pong_step.onnx      物理引擎（10.8K）
│       └── pong_policy.onnx    AI 决策（2.7K）
├── package.json                npm 依赖
└── node_modules/               onnxruntime-web（npm install 后生成）
```
