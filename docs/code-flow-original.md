# Code Flow — 原始版本 (src/original)

纯 JS 实现，所有逻辑在 `script.js` 中同步执行。

---

## 1. 启动流程

```
浏览器加载 index.html
  └─ <script src="script.js">
       ├─ 解析全部代码（同步，瞬时完成）
       ├─ 注册 DOM 引用（canvas, overlay, buttons...）
       ├─ 定义函数（clamp, lerp, rand, aiUpdate, draw...）
       ├─ 注册事件监听器（click, keydown, pointermove...）
       ├─ 执行初始化代码（L776-791）：
       │    resize()
       │    paddle.y = H / 2
       │    ai.y = H / 2
       │    overlayImg.src = INTRO_IMG
       │    playText = "START"
       └─ requestAnimationFrame(update)  ← 游戏循环开始
```

**耗时**: 毫秒级，无异步等待，用户立即看到 "START"。

---

## 2. 游戏循环 (`update` — 每帧 ~16ms)

```
update(ts)                                           ← requestAnimationFrame 回调
│
├─ requestAnimationFrame(update)                     排队下一帧
│
├─ dt 计算
│   dt = clamp(raw_dt, 0.008, 0.02)                  JS 内联 Math.min/Math.max
│   dt *= state.slowmo
│
├─ 全局状态更新
│   state.time += dt
│   state.shake = Math.max(0, shake - dt * 22)       JS 内联
│
├─ updateCutscene(dt)                                JS 函数调用
│   └─ lerp() / Math.pow()                           JS 内联数学
│
├─ 键盘输入
│   paddle.targetY = clamp(...)                      JS 函数调用
│
├─ if (!running || paused) → draw() → return
│
├─ 玩家球拍平滑
│   paddle.y = lerp(paddle.y, targetY, ease)         JS 函数调用
│
├─ aiUpdate(dt)                                      JS 函数调用
│   ├─ levelFromProgress()                           JS 函数调用
│   ├─ 查表 [0.24, 0.2, ...][lvl]                    JS 数组索引
│   ├─ lerp() / clamp() / Math.pow()                 JS 函数调用
│   └─ ai.y += ai.vy * dt
│
├─ 球移动
│   ball.x += ball.vx * dt                           JS 内联
│   ball.y += ball.vy * dt                           JS 内联
│
├─ 拖尾
│   trail.push({ x, y, t })                          JS 数组 push
│
├─ 墙壁碰撞
│   if (ball.y - ball.r <= 0) {                      JS if 分支
│     ball.y = ball.r; ball.vy *= -1
│     sparkLine(...); beep("wall")
│   }
│   (同理 底部墙壁)
│
├─ 玩家球拍碰撞
│   if (ball.vx < 0 && ball.x-ball.r <= px+w/2) {    JS 嵌套 if
│     if (ball.y in [top, bot]) {
│       位置修正 / 角度计算 / 速度缩放               JS 内联数学
│       burst(); sparkLine(); shock()                JS 函数调用
│       beep("hit")
│     }
│   }
│
├─ AI 球拍碰撞                                       (同上，镜像)
│
├─ 得分检测
│   if (ball.x < -60) {                              JS if 分支
│     score.R++; rally = 0
│     burst(); shock(); beep("score")
│     checkWinOrReset(-1)
│       └─ resetRound() 或 endGame()
│   }
│   (同理 ball.x > W+60)
│
├─ updateParticles(dt)                               JS 函数调用
│   ├─ for 循环遍历 particles[]                      JS 对象数组
│   │   p.x += p.vx * dt; p.vx *= decay
│   │   if (p.t >= p.life) splice(i, 1)              JS 数组 splice
│   └─ for 循环遍历 sparks[]                         (同上)
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
        ├─ for (p of particles) fillRect             遍历 JS 对象数组
        └─ for (p of sparks) moveTo/lineTo           遍历 JS 对象数组
```

---

## 3. 用户交互

```
鼠标/触摸移动
  └─ setTargetFromClientY(clientY)                   同步
       paddle.targetY = clamp(y, min, max)           JS 函数调用（立即返回）

键盘按下
  ├─ Space → start() 或 togglePause()                同步
  ├─ R → hardRestart()                               同步
  └─ M → toggleSound()                               同步

按钮点击
  ├─ START → start()                                 同步
  │   └─ resetGame() → resetRound()
  ├─ PAUSE → togglePause()                           同步
  ├─ SOUND → toggleSound()                           同步
  └─ FX → toggleFX()                                 同步
```

---

## 4. 数据流

```
输入层           计算层                              输出层
─────────       ─────────                           ─────────
鼠标 Y ────────→ paddle.targetY → lerp → paddle.y ─→ drawPaddle()
键盘 W/S ──────→ paddle.targetY → lerp → paddle.y ─→ drawPaddle()
ball state ────→ aiUpdate() ───────────────────────→ ai.y → drawPaddle()
ball state ────→ ball.x/y += v*dt ─────────────────→ drawBall()
ball + paddle ─→ 碰撞检测 ────────────────────────→  events
events ────────→ burst() / sparkLine() ────────────→ particles[] → drawParticles()
events ────────→ beep() ───────────────────────────→ Web Audio
events ────────→ score++ ──────────────────────────→ DOM 更新
```

---

## 5. 关键特征

| 特征     | 说明                                            |
| -------- | ----------------------------------------------- |
| 执行模型 | **全同步**，单线程                              |
| 启动时间 | 毫秒级（无异步加载）                            |
| 函数调用 | 直接 JS 函数调用，零开销                        |
| 粒子存储 | JS 对象数组 `particles[]`，`push`/`splice` 管理 |
| 分支逻辑 | `if/else` 语句                                  |
| 数学计算 | JS 内联 `Math.*` + 自定义 `clamp`/`lerp`/`rand` |
| 外部依赖 | Google Fonts, Font Awesome, 图片（CDN）         |
| 运行方式 | `file://` 直接打开即可                          |
