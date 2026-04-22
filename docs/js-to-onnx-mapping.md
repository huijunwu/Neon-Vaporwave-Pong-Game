# JS 函数与 ONNX 映射表

源码 JS: `src/original/script.js`（原始版参考）
当前 JS: `dist/js/script.js`
Python models: `src/python/pong/onnx_modules.py`
ONNX 输出: `dist/assets/onnx/`

原则：**物理引擎 + AI 决策 → ONNX，其余留 JS**

---

## 总览

| 分类         | 数量   | ONNX  | JS 原生 |
| ------------ | ------ | ----- | ------- |
| 游戏物理     | 1      | 1     | 0       |
| AI 决策      | 1      | 1     | 0       |
| 数学工具     | 3      | 0     | 3       |
| 视觉特效     | 4      | 0     | 4       |
| UI / DOM     | 10     | 0     | 10      |
| 音频         | 3      | 0     | 3       |
| 渲染         | 5      | 0     | 5       |
| 游戏生命周期 | 6      | 0     | 6       |
| **合计**     | **33** | **2** | **31**  |

---

## ONNX 替换的函数

| #   | 原始 JS 逻辑                                         | ONNX 模型          | Python 类                      | 说明                                                   |
| --- | ---------------------------------------------------- | ------------------ | ------------------------------ | ------------------------------------------------------ |
| 1   | `aiUpdate(dt)` + `levelFromProgress()`               | `pong_policy.onnx` | `PongPolicy` (onnx_modules.py) | AI 决策：obs → action (0/1/2) + memory。双方各调用一次 |
| 2   | `update()` 物理块（球拍移动 + 球移动 + 碰撞 + 得分） | `pong_step.onnx`   | `PongStep` (onnx_modules.py)   | 接受 W/H 参数，适配任意 Canvas 尺寸                    |

## JS 原生保留的函数

### 数学工具

| #   | JS 函数          | 说明                          |
| --- | ---------------- | ----------------------------- |
| 3   | `clamp(v, a, b)` | `Math.max(a, Math.min(b, v))` |
| 4   | `lerp(a, b, t)`  | `a + (b - a) * t`             |
| 5   | `rand(a, b)`     | `a + Math.random() * (b - a)` |

### 视觉特效

| #   | JS 函数                      | 说明                                     |
| --- | ---------------------------- | ---------------------------------------- |
| 6   | `shock(a)`                   | `min(10, shake + a)`                     |
| 7   | `updateCutscene(dt)`         | slowmo 过渡逻辑                          |
| 8   | `updateParticles(dt)`        | 粒子/火花位置更新（Float32Array 缓冲区） |
| 9   | `burst(x, y, n)`             | 生成粒子（使用 `rand()`）                |
| 10  | `sparkLine(x, y, vx, vy, n)` | 生成火花（使用 `rand()`）                |

### UI / DOM（浏览器 API，无法 ONNX）

| #   | JS 函数                    | 说明                     |
| --- | -------------------------- | ------------------------ |
| 11  | `setPressed(el, on)`       | DOM：设置 `aria-pressed` |
| 12  | `setIntro(on)`             | DOM：classList toggle    |
| 13  | `showOverlay(title, body)` | DOM：显示遮罩层          |
| 14  | `hideOverlay()`            | DOM：隐藏遮罩层          |
| 15  | `setSoundIcon()`           | DOM：切换图标 class      |
| 16  | `setFXIcon()`              | DOM：切换图标 class      |
| 17  | `setPauseIcon()`           | DOM：切换图标 class      |
| 18  | `togglePause()`            | DOM + 状态机             |
| 19  | `toggleSound()`            | DOM + 状态切换           |
| 20  | `toggleFX()`               | DOM + 状态切换           |

### 音频（Web Audio API，无法 ONNX）

| #   | JS 函数                       | 说明                    |
| --- | ----------------------------- | ----------------------- |
| 21  | `ensureAudio()`               | 创建 AudioContext       |
| 22  | `tone(freq, dur, type, gain)` | OscillatorNode 合成音效 |
| 23  | `beep(kind)`                  | 音效事件分发器          |

### 渲染（Canvas 2D API，无法 ONNX）

| #   | JS 函数                            | 说明        |
| --- | ---------------------------------- | ----------- |
| 24  | `roundRect(x, y, w, h, r)`         | Canvas 路径 |
| 25  | `draw()`                           | 主渲染函数  |
| 26  | `drawPaddle(x, y, w, h, isPlayer)` | Canvas 绘制 |
| 27  | `drawBall(x, y, r)`                | Canvas 绘制 |
| 28  | `drawParticles()`                  | Canvas 绘制 |

### 游戏生命周期（状态机 + DOM + `Math.random()`，无法 ONNX）

| #   | JS 函数                    | 说明                                    |
| --- | -------------------------- | --------------------------------------- |
| 29  | `resetRound(direction)`    | `Math.random()` + 副作用（burst, beep） |
| 30  | `resetGame()`              | 重置所有状态 + DOM                      |
| 31  | `start()` / `hardRestart`  | 状态机入口 + DOM（hardRestart = start） |
| 32  | `checkWinOrReset(nextDir)` | 判定得分 → endGame 或 resetRound        |
| 33  | `endGame(playerWon)`       | 状态机 + DOM 遮罩层                     |

---

## 2 个 ONNX 模型接口详情

### pong_policy.onnx（AI 决策）

```
输入:
  obs         float32[6]   归一化观测 [ball_x/W, ball_y/H, ball_vx/S, ball_vy/S, own_y/H, opp_y/H]
  memory_y    float32      AI 平滑目标记忆值
  rand_val    float32      随机值 [-1, 1]
  H           float32      Canvas 高度

输出:
  action        float32    离散动作 (0.0=NOOP, 1.0=DOWN, 2.0=UP)
  new_memory_y  float32    更新后记忆值
```

### pong_step.onnx（物理引擎）

```
输入:
  ball_x          float32    球 X 坐标
  ball_y          float32    球 Y 坐标
  ball_vx         float32    球 X 速度
  ball_vy         float32    球 Y 速度
  paddle_left_y   float32    左球拍 Y 坐标
  paddle_right_y  float32    右球拍 Y 坐标
  action_left     float32    左球拍动作 (0/1/2)
  action_right    float32    右球拍动作 (0/1/2)
  rally           float32    当前回合击球数
  W               float32    Canvas 宽度
  H               float32    Canvas 高度

输出:
  new_ball_x          float32      更新后球 X 坐标
  new_ball_y          float32      更新后球 Y 坐标
  new_ball_vx         float32      更新后球 X 速度
  new_ball_vy         float32      更新后球 Y 速度
  new_paddle_left_y   float32      更新后左球拍 Y
  new_paddle_right_y  float32      更新后右球拍 Y
  new_rally           float32      更新后击球数
  events              float32[6]   [hit_left, hit_right, wall_top, wall_bottom, scored_left, scored_right]
```

所有输入输出统一 float32，无 int64 边界。

---

## 每帧调用顺序

```
JS update(ts):
  1.  dt = clamp(raw_dt, 0.008, 0.02) * slowmo                           ← JS 原生
  2.  state.time += dt; state.shake 衰减                                 ← JS 原生
  3.  updateCutscene(dt)                                                 ← JS 原生
  4.  if (!running || paused) → draw() → return
  5.  obsL = buildObs(true); obsR = buildObs(false)                      ← JS 构建观测
  6.  leftResult = await onnxPolicy(obsL, memoryL, rand, H)              ☆ ONNX policy
  7.  rightResult = await onnxPolicy(obsR, memoryR, rand, H)             ☆ ONNX policy
  8.  stepResult = await onnxStep(ball, paddles, actions, rally, W, H)   ☆ ONNX step
      → 更新: ball, paddles, rally + events[6]
  9.  拖尾管理                                                           ← JS
  10. 根据 events 触发副作用:
        墙壁碰撞 → sparkLine() + beep()                                  ← JS
        球拍碰撞 → burst() + sparkLine() + shock() + beep()              ← JS
        得分 → handleScore() → checkWinOrReset()                         ← JS
  11. updateParticles(dt)                                                ← JS 原生
  12. draw()                                                             ← JS Canvas
```
