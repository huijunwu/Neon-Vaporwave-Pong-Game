# JS 函数与 ONNX 映射表

源码 JS: `src/original/script.js`
Python models: `src/python/models.py`
ONNX 输出: `dist/onnx/`

原则：**纯数学函数 → 全部转 ONNX，只有浏览器 API 留 JS**

---

## 总览

| 分类         | 数量   | ONNX 替换 | 保留在 JS |
| ------------ | ------ | --------- | --------- |
| 游戏逻辑     | 5      | 4         | 1         |
| 数学工具     | 3      | 2         | 1         |
| UI / DOM     | 10     | 0         | 10        |
| 音频         | 3      | 0         | 3         |
| 输入         | 1      | 0         | 1         |
| 视觉特效     | 4      | 1         | 3         |
| 渲染         | 5      | 0         | 5         |
| 游戏生命周期 | 6      | 0         | 6         |
| **合计**     | **37** | **7**     | **30**    |

---

## 完整函数清单

### 数学工具

| #   | JS 函数 (行号)         | ONNX?   | Python 类     | ONNX 文件         | 说明                                        |
| --- | ---------------------- | ------- | ------------- | ----------------- | ------------------------------------------- |
| 1   | `clamp(v, a, b)` (L42) | **YES** | `Clamp` (L49) | `pong_clamp.onnx` | `max(a, min(v, b))`                         |
| 2   | `lerp(a, b, t)` (L45)  | **YES** | `Lerp` (L58)  | `pong_lerp.onnx`  | `a + (b - a) * t`                           |
| 3   | `rand(a, b)` (L39)     | no      | —             | —                 | 核心是 `Math.random()`，ONNX 无法生成随机数 |

### 游戏逻辑（核心计算）

| #   | JS 函数 (行号)                     | ONNX?   | Python 类                         | ONNX 文件              | 说明                                            |
| --- | ---------------------------------- | ------- | --------------------------------- | ---------------------- | ----------------------------------------------- |
| 4   | `levelFromProgress()` (L286)       | **YES** | `level_from_progress()` (L21)     | (内联到 ai 和 physics) | 共享辅助函数，内联到 AIUpdate 和 BallPhysics 中 |
| 5   | `aiUpdate(dt)` (L399)              | **YES** | `AIUpdate.forward()` (L82)        | `pong_ai.onnx`         | AI 球拍目标预测 + 平滑移动                      |
| 6   | `update(ts)` 物理块 (L622-L738)    | **YES** | `BallPhysics.forward()` (L130)    | `pong_physics.onnx`    | 球拍平滑 + 球移动 + 碰撞 + 得分检测             |
| 7   | `updateCutscene(dt)` (L319)        | **YES** | `CutsceneUpdate.forward()` (L245) | `pong_cutscene.onnx`   | slowmo 过渡逻辑                                 |
| 8   | `maybeMatchPointCutscene()` (L297) | no      | —                                 | —                      | sub-func 级别条件判断 + DOM 操作，块不大，留 JS |

### 视觉特效

| #   | JS 函数 (行号)                      | ONNX?   | Python 类               | ONNX 文件             | 说明                                         |
| --- | ----------------------------------- | ------- | ----------------------- | --------------------- | -------------------------------------------- |
| 9   | `shock(a)` (L282)                   | **YES** | `Shock` (L67)           | `pong_shock.onnx`     | `min(10, shake + a)`                         |
| 10  | `updateParticles(dt)` (L425)        | **YES** | `ParticleUpdate` (L285) | `pong_particles.onnx` | 固定大小 tensor [70,7] / [40,7] + alive 掩码 |
| 11  | `burst(x, y, n)` (L254)             | no      | —                       | —                     | 使用 `rand()` + 数组 `push()`，浏览器 API    |
| 12  | `sparkLine(x, y, vx, vy, n)` (L268) | no      | —                       | —                     | 使用 `rand()` + 数组 `push()`，浏览器 API    |

### UI / DOM（浏览器 API，无法 ONNX）

| #   | JS 函数 (行号)                   | 原因                     |
| --- | -------------------------------- | ------------------------ |
| 13  | `setPressed(el, on)` (L78)       | DOM：设置 `aria-pressed` |
| 14  | `setIntro(on)` (L81)             | DOM：classList toggle    |
| 15  | `showOverlay(title, body)` (L85) | DOM：显示遮罩层          |
| 16  | `hideOverlay()` (L92)            | DOM：隐藏遮罩层          |
| 17  | `setSoundIcon()` (L96)           | DOM：切换图标 class      |
| 18  | `setFXIcon()` (L104)             | DOM：切换图标 class      |
| 19  | `setPauseIcon()` (L112)          | DOM：切换图标 class      |
| 20  | `togglePause()` (L135)           | DOM + 状态机             |
| 21  | `toggleSound()` (L151)           | DOM + 状态切换           |
| 22  | `toggleFX()` (L158)              | DOM + 状态切换           |

### 音频（Web Audio API，无法 ONNX）

| #   | JS 函数 (行号)                       | 原因                    |
| --- | ------------------------------------ | ----------------------- |
| 23  | `ensureAudio()` (L171)               | 创建 AudioContext       |
| 24  | `tone(freq, dur, type, gain)` (L178) | OscillatorNode 合成音效 |
| 25  | `beep(kind)` (L194)                  | 音效事件分发器          |

### 输入（DOM 事件，无法 ONNX）

| #   | JS 函数 (行号)                         | 原因                      |
| --- | -------------------------------------- | ------------------------- |
| 26  | `setTargetFromClientY(clientY)` (L211) | DOM 事件 → paddle targetY |

### 渲染（Canvas 2D API，无法 ONNX）

| #   | JS 函数 (行号)                            | 原因        |
| --- | ----------------------------------------- | ----------- |
| 27  | `roundRect(x, y, w, h, r)` (L446)         | Canvas 路径 |
| 28  | `draw()` (L457)                           | 主渲染函数  |
| 29  | `drawPaddle(x, y, w, h, isPlayer)` (L511) | Canvas 绘制 |
| 30  | `drawBall(x, y, r)` (L535)                | Canvas 绘制 |
| 31  | `drawParticles()` (L561)                  | Canvas 绘制 |

### 游戏生命周期（状态机 + DOM + `Math.random()`，无法 ONNX）

| #   | JS 函数 (行号)                    | 原因                                    |
| --- | --------------------------------- | --------------------------------------- |
| 32  | `resetRound(direction)` (L334)    | `Math.random()` + 副作用（burst, beep） |
| 33  | `resetGame()` (L353)              | 重置所有状态 + DOM                      |
| 34  | `start()` (L377)                  | 状态机入口 + DOM                        |
| 35  | `hardRestart()` (L388)            | 状态机入口 + DOM                        |
| 36  | `checkWinOrReset(nextDir)` (L744) | 判定得分 → endGame 或 resetRound        |
| 37  | `endGame(playerWon)` (L756)       | 状态机 + DOM 遮罩层                     |

---

## 7 个 ONNX 模型接口详情

### pong_clamp.onnx（替换 `clamp`）

```
输入:  v float32, a float32, b float32
输出:  result float32
```

### pong_lerp.onnx（替换 `lerp`）

```
输入:  a float32, b float32, t float32
输出:  result float32
```

### pong_shock.onnx（替换 `shock`）

```
输入:  shake float32 (当前震动值), amount float32 (新增量)
输出:  new_shake float32
```

### pong_ai.onnx（替换 `aiUpdate` + `levelFromProgress`）

```
输入:
  ball_y        float32    球 Y 坐标
  ball_vy       float32    球 Y 速度
  ball_vx       float32    球 X 速度
  ai_y          float32    AI 球拍 Y 坐标
  ai_memoryY    float32    AI 平滑目标记忆值
  score_L       float32    玩家得分
  score_R       float32    AI 得分
  dt            float32    帧间隔（秒）
  H             float32    Canvas 高度
  rand_val      float32    随机值 [-1, 1]

输出:
  new_ai_y       float32   更新后 AI 球拍 Y 坐标
  new_ai_memoryY float32   更新后记忆值
  ai_vy          float32   AI 球拍速度
  ai_h           float32   AI 球拍高度（随难度缩小）
```

### pong_physics.onnx（替换 `update()` 物理块 + `levelFromProgress`）

```
输入:
  ball_x          float32    球 X 坐标
  ball_y          float32    球 Y 坐标
  ball_vx         float32    球 X 速度
  ball_vy         float32    球 Y 速度
  paddle_y        float32    玩家球拍 Y 坐标
  paddle_target_y float32    玩家球拍目标 Y（来自输入）
  paddle_h        float32    玩家球拍高度
  ai_y            float32    AI 球拍 Y 坐标
  ai_h            float32    AI 球拍高度（来自 pong_ai.onnx）
  score_L         float32    玩家得分
  score_R         float32    AI 得分
  rally           float32    当前回合击球数
  dt              float32    帧间隔（秒）
  W               float32    Canvas 宽度
  H               float32    Canvas 高度

输出:
  new_ball_x    float32      更新后球 X 坐标
  new_ball_y    float32      更新后球 Y 坐标
  new_ball_vx   float32      更新后球 X 速度
  new_ball_vy   float32      更新后球 Y 速度
  new_paddle_y  float32      更新后玩家球拍 Y（已平滑）
  new_rally     float32      更新后击球数
  events        float32[6]   [hit_player, hit_ai, wall_top, wall_bottom, scored_L, scored_R]
```

### pong_cutscene.onnx（替换 `updateCutscene`）

```
输入:
  cut       float32    过场动画激活标记（0.0 或 1.0）
  cutT      float32    过场动画已经过时间
  slowmo    float32    当前慢动作系数
  dt        float32    帧间隔（秒）

输出:
  new_cut     float32    更新后过场动画标记
  new_cutT    float32    更新后已经过时间
  new_slowmo  float32    更新后慢动作系数
  cut_ended   float32    1.0 表示过场动画刚结束（JS 需移除 CSS class）
```

### pong_particles.onnx（替换 `updateParticles`）

```
输入:
  particles   float32[70, 7]   粒子缓冲区 [x, y, vx, vy, life, t, alive]
  sparks      float32[40, 7]   火花缓冲区 [x, y, vx, vy, life, t, alive]
  dt          float32           帧间隔（秒）

输出:
  new_particles  float32[70, 7]  更新后粒子缓冲区
  new_sparks     float32[40, 7]  更新后火花缓冲区

说明:
  - alive=1.0 表示存活，alive=0.0 表示空槽位
  - 粒子速度衰减系数 0.14^dt，火花衰减系数 0.06^dt
  - 超过 life 时间的粒子自动标记为 alive=0.0
  - JS 负责通过 alive 掩码管理新增（burst/sparkLine）和渲染
```

---

## 每帧调用顺序

```
JS update(ts):
  1.  dt = onnx.clamp(raw_dt, 0.008, 0.02)                             ← ONNX
      dt *= state.slowmo                                               ← JS（单次乘法）
  2.  state.time += dt                                                 ← JS（单次加法）
      state.shake = Math.max(0, state.shake - dt * 22)                 ← JS（衰减，sub-func 级别）
  3.  onnx.cutscene_update(cut, cutT, slowmo, dt)                      ← ONNX
      → 若 cut_ended: JS 移除过场动画 CSS class
  4.  键盘输入 → paddle.targetY（使用 onnx.clamp 限制范围）            ← ONNX + JS
  5.  if (!running || paused) { draw(); return; }                      ← JS（状态检查）
  6.  onnx.ai_update(ball, ai, scores, dt, H, rand)                    ← ONNX
      → 更新: ai.y, ai.memoryY, ai.vy, ai.h
  7.  onnx.ball_physics(ball, paddle, ai, scores, rally, dt, W, H)     ← ONNX
      → 更新: ball.x/y/vx/vy, paddle.y, rally
      → events: JS 检查标记触发副作用
  8.  若 hit_player/hit_ai:
        state.shake = onnx.shock(state.shake, 4)                       ← ONNX
        burst(), sparkLine(), beep()                                   ← JS（rand + DOM + 音频）
  9.  若 wall_top/wall_bottom: sparkLine(), beep()                     ← JS（副作用）
  10. 若 scored_L/scored_R:
        score++, rally=0                                               ← JS
        state.shake = onnx.shock(state.shake, 6)                       ← ONNX
        checkWinOrReset()                                              ← JS（状态机）
  11. onnx.particle_update(particles, sparks, dt)                      ← ONNX
  12. draw()                                                           ← JS（Canvas API）
```
