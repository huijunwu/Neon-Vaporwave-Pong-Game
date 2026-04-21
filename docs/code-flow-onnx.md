# Code Flow — ONNX 版本 (dist)

JS + ONNX Runtime Web 实现，纯数学逻辑由 7 个 ONNX 模型执行。

---

## 1. 启动流程

```
浏览器加载 index.html
  ├─ <script src="onnxruntime-web CDN">              加载 ort 运行时
  └─ <script src="script.js">
       ├─ 解析全部代码（同步）
       ├─ 注册 DOM 引用（canvas, overlay, buttons...）
       ├─ 定义 ONNX wrapper 函数（clamp, lerp, onnxShock...）
       ├─ 注册事件监听器（click, keydown, pointermove...）
       └─ init()                                      async 入口
            ├─ resize() / 初始化状态
            ├─ overlayImg.src = INTRO_IMG
            ├─ playText = "LOADING…"                  ← 加载提示
            ├─ await loadONNX()                       异步加载 7 个 ONNX 模型
            │   ├─ pong_clamp.onnx      (359B)
            │   ├─ pong_lerp.onnx       (380B)
            │   ├─ pong_shock.onnx      (417B)
            │   ├─ pong_ai.onnx         (7.7K)
            │   ├─ pong_physics.onnx    (12K)
            │   ├─ pong_cutscene.onnx   (1.8K)
            │   └─ pong_particles.onnx  (5.9K)
            ├─ playText = "START"                     ← 加载完成
            └─ requestAnimationFrame(update)          游戏循环开始
```

**耗时**: 数百毫秒（ONNX 模型 fetch + WASM 初始化），用户先看到 "LOADING…" 再变 "START"。

---

## 2. 游戏循环 (`update` — 每帧 ~16ms)

```
async update(ts)                                     ← requestAnimationFrame 回调
│
├─ requestAnimationFrame(update)                     排队下一帧
│
├─ dt 计算
│   dt = await clamp(raw_dt, 0.008, 0.02)            ☆ ONNX: pong_clamp.onnx
│   dt *= state.slowmo
│
├─ 全局状态更新
│   state.time += dt
│   state.shake = Math.max(0, shake - dt * 22)       JS 内联
│
├─ await onnxCutscene(dt)                            ☆ ONNX: pong_cutscene.onnx
│   └─ 返回: new_cut, new_cutT, new_slowmo, cut_ended
│   if (cut_ended) → JS 移除 CSS class
│
├─ 键盘输入
│   paddle.targetY = await clamp(...)                ☆ ONNX: pong_clamp.onnx
│
├─ if (!running || paused) → draw() → return
│
├─ await onnxAiUpdate(dt)                            ☆ ONNX: pong_ai.onnx
│   ├─ 输入: ball 状态, ai 状态, scores, dt, H, rand_val
│   │   (rand_val = Math.random()*2-1，JS 生成)
│   └─ 输出 → ai.y, ai.memoryY, ai.vy, ai.h
│
├─ events = await onnxPhysics(dt)                    ☆ ONNX: pong_physics.onnx
│   ├─ 输入: ball, paddle, ai, scores, rally, dt, W, H
│   ├─ 内部执行:
│   │   ├─ 玩家球拍平滑 (lerp)
│   │   ├─ 球移动 (x += vx*dt)
│   │   ├─ 墙壁碰撞 (torch.where)
│   │   ├─ 玩家球拍碰撞 (torch.where)
│   │   ├─ AI 球拍碰撞 (torch.where)
│   │   └─ 得分检测
│   └─ 输出 → ball.x/y/vx/vy, paddle.y, rally, events[6]
│
├─ 拖尾
│   trail.push({ x, y, t })                          JS 数组 push
│
├─ 处理 events（来自 ONNX 的 float32[6] 标记）
│   ├─ wallTop/wallBottom:
│   │   sparkLine(); beep("wall")                    JS（rand + Canvas + Audio）
│   ├─ hitPlayer/hitAi:
│   │   burst(); sparkLine()                         JS（rand + Canvas）
│   │   await onnxShock(4)                           ☆ ONNX: pong_shock.onnx
│   │   beep("hit")                                  JS（Audio）
│   └─ scoredL/scoredR:
│       score++; rally = 0                           JS
│       burst()                                      JS（rand + Canvas）
│       await onnxShock(6)                           ☆ ONNX: pong_shock.onnx
│       beep("score")                                JS（Audio）
│       await checkWinOrReset()
│         └─ resetRound() 或 endGame()               JS（rand + DOM）
│
├─ await onnxParticles(dt)                           ☆ ONNX: pong_particles.onnx
│   ├─ 输入: particleBuf[70×7], sparkBuf[40×7], dt
│   ├─ 内部执行:
│   │   ├─ 位置更新 (x += vx*dt)
│   │   ├─ 速度衰减 (vx *= 0.14^dt / 0.06^dt)
│   │   └─ 生命周期 (alive = alive * (t < life))
│   └─ 输出 → particleBuf, sparkBuf（原地覆写）
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
鼠标/触摸移动
  └─ async setTargetFromClientY(clientY)             异步
       paddle.targetY = await clamp(y, min, max)     ☆ ONNX（微秒级延迟，不影响体验）

键盘按下
  ├─ Space → start() 或 togglePause()                async
  ├─ R → hardRestart()                               async
  └─ M → toggleSound()                               同步

按钮点击
  ├─ START → start()                                 async
  │   └─ resetGame() → resetRound()
  │       └─ onnxShock(3)                            ☆ ONNX
  ├─ PAUSE → togglePause()                           同步
  ├─ SOUND → toggleSound()                           同步
  └─ FX → toggleFX()                                 同步
       └─ particleBuf.fill(0); sparkBuf.fill(0)      清空固定缓冲区
```

---

## 4. 数据流

```
输入层              计算层                              输出层
─────────          ─────────                           ─────────
鼠标 Y ───────────→ ☆ ONNX clamp → paddle.targetY ───→  (存入 JS 变量)
键盘 W/S ─────────→ ☆ ONNX clamp → paddle.targetY ───→  (存入 JS 变量)
JS 变量 → tensor ─→ ☆ ONNX ai → 解包 ───────────────→   ai.y → drawPaddle()
JS 变量 → tensor ─→ ☆ ONNX physics → 解包 ──────────→   ball/paddle → draw()
events[6] ────────→ burst() / sparkLine() ────────────→ particleBuf → drawParticles()
events[6] ────────→ beep() ───────────────────────────→ Web Audio
events[6] ────────→ ☆ ONNX shock ────────────────────→  state.shake
events[6] ────────→ score++ ──────────────────────────→ DOM 更新
particleBuf ──────→ ☆ ONNX particles → 覆写 ─────────→  particleBuf → drawParticles()
sparkBuf ─────────→ ☆ ONNX particles → 覆写 ─────────→  sparkBuf → drawParticles()
```

---

## 5. 关键特征

| 特征     | 说明                                                              |
| -------- | ----------------------------------------------------------------- |
| 执行模型 | **async/await**，ONNX 推理通过 WASM 后端                          |
| 启动时间 | 数百毫秒（需加载 7 个 .onnx + WASM runtime）                      |
| 函数调用 | JS → `session.run()` → WASM → 返回 tensor → JS 解包               |
| 粒子存储 | `Float32Array` 固定缓冲区 [70×7] / [40×7]，alive 掩码管理         |
| 分支逻辑 | ONNX 内部用 `Where` 算子（等价 `torch.where`）                    |
| 数学计算 | ONNX 计算图（Pow, Clamp, Sqrt, Where...）                         |
| 外部依赖 | Google Fonts, Font Awesome, 图片（本地），onnxruntime-web（CDN）  |
| 运行方式 | **必须 HTTP 服务器**（`python -m http.server`），不支持 `file://` |
