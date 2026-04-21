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
├── index.html          入口页面
├── script.js           游戏主线程逻辑
├── onnx-worker.mjs     ONNX 推理 Worker（ES module）
├── style.css           样式
├── package.json        npm 依赖（onnxruntime-web）
├── assets/             图片资源
│   ├── codepong-title.png
│   └── codepong26.png
├── node_modules/       npm install 后生成（.gitignore）
│   └── onnxruntime-web/  WASM 运行时 + ONNX Runtime
└── onnx/               ONNX 模型（7 个）
    ├── pong_clamp.onnx
    ├── pong_lerp.onnx
    ├── pong_shock.onnx
    ├── pong_ai.onnx
    ├── pong_physics.onnx
    ├── pong_cutscene.onnx
    └── pong_particles.onnx
```

`onnxruntime-web` 通过 npm 安装到本地 `node_modules/`，Worker 以 ES module 方式 import。
