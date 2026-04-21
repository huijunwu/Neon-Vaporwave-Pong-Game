/* ═══════════════════════════════════════════════════════════════════
   ONNX Worker — all inference runs off the main thread
   ═══════════════════════════════════════════════════════════════════ */

const worker = new Worker("./onnx-worker.mjs", { type: "module" });
let _reqId = 0;
const _pending = new Map();

worker.onmessage = (e) => {
	const { type, id, outputs, message } = e.data;
	if (type === "ready") {
		_pending.get("init")?.resolve();
		_pending.delete("init");
	}
	if (type === "result") {
		_pending.get(id)?.resolve(outputs);
		_pending.delete(id);
	}
	if (type === "error") {
		const p = _pending.get(id ?? "init");
		if (p) {
			p.reject(new Error(message));
			_pending.delete(id ?? "init");
		}
	}
};

worker.onerror = (e) => {
	console.error("ONNX Worker error:", e.message);
};

function workerInit() {
	return new Promise((resolve, reject) => {
		_pending.set("init", { resolve, reject });
		worker.postMessage({
			type: "init",
			base: "./onnx/",
			models: ["clamp", "lerp", "shock", "ai", "physics", "cutscene", "particles"]
		});
	});
}

function workerRun(model, inputs) {
	return new Promise((resolve, reject) => {
		const id = _reqId++;
		_pending.set(id, { resolve, reject });
		worker.postMessage({ id, type: "run", model, inputs });
	});
}

/* ═══════════════════════════════════════════════════════════════════
   DOM references
   ═══════════════════════════════════════════════════════════════════ */

const canvas = document.getElementById("c");
const ctx = canvas.getContext("2d", { alpha: true });

const overlay = document.getElementById("overlay");
const overlayImg = document.getElementById("overlayImg");

const btnPlay = document.getElementById("btnPlay");
const playText = document.getElementById("playText");
const btnHow = document.getElementById("btnHow");
const btnPause = document.getElementById("btnPause");
const btnSound = document.getElementById("btnSound");
const btnFX = document.getElementById("btnFX");

const cutsceneEl = document.getElementById("cutscene");
const cutText = document.getElementById("cutText");
const cutSub = document.getElementById("cutSub");

const sL = document.getElementById("sL");
const sR = document.getElementById("sR");

const INTRO_IMG = "./assets/codepong-title.png";
const END_IMG = "./assets/codepong26.png";

/* ═══════════════════════════════════════════════════════════════════
   State / constants
   ═══════════════════════════════════════════════════════════════════ */

let W = 0,
	H = 0;
let DPR = 1;

const state = {
	running: false,
	paused: false,
	fx: true,
	sound: false,
	time: 0,
	shake: 0,
	cut: false,
	cutT: 0,
	slowmo: 1
};

const score = { L: 0, R: 0, toWin: 11 };
let rally = 0;

const paddle = { w: 14, h: 110, inset: 26, y: 0, targetY: 0 };
const ai = { w: 14, h: 110, inset: 26, y: 0, vy: 0, memoryY: 0 };
const ball = { r: 10, x: 0, y: 0, vx: 0, vy: 0, speed: 560 };

const trail = [];

const MAX_P = 70;
const MAX_S = 40;
const particleBuf = new Float32Array(MAX_P * 7); // [x,y,vx,vy,life,t,alive]
const sparkBuf = new Float32Array(MAX_S * 7);

const PERF = { maxTrail: 16 };

/* ═══════════════════════════════════════════════════════════════════
   Helpers — rand stays in JS (Math.random)
   ═══════════════════════════════════════════════════════════════════ */

function rand(a, b) {
	return a + Math.random() * (b - a);
}

function resize() {
	const rect = canvas.getBoundingClientRect();
	W = Math.floor(rect.width);
	H = Math.floor(rect.height);
	canvas.width = W * DPR;
	canvas.height = H * DPR;
	ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
}
window.addEventListener("resize", resize);

/* ═══════════════════════════════════════════════════════════════════
   ONNX wrapper functions
   ═══════════════════════════════════════════════════════════════════ */

async function clamp(v, a, b) {
	const r = await workerRun("clamp", { v, a, b });
	return r.result;
}

async function lerp(a, b, t) {
	const r = await workerRun("lerp", { a, b, t });
	return r.result;
}

async function onnxShock(amount) {
	const r = await workerRun("shock", { shake: state.shake, amount });
	state.shake = r.new_shake;
}

async function onnxAiUpdate(dt) {
	const r = await workerRun("ai", {
		ball_y: ball.y,
		ball_vy: ball.vy,
		ball_vx: ball.vx,
		ai_y: ai.y,
		ai_memoryY: ai.memoryY,
		score_L: score.L,
		score_R: score.R,
		dt,
		H,
		rand_val: Math.random() * 2 - 1
	});
	ai.y = r.new_ai_y;
	ai.memoryY = r.new_ai_memoryY;
	ai.vy = r.ai_vy;
	ai.h = r.ai_h;
}

async function onnxPhysics(dt) {
	const r = await workerRun("physics", {
		ball_x: ball.x,
		ball_y: ball.y,
		ball_vx: ball.vx,
		ball_vy: ball.vy,
		paddle_y: paddle.y,
		paddle_target_y: paddle.targetY,
		paddle_h: paddle.h,
		ai_y: ai.y,
		ai_h: ai.h,
		score_L: score.L,
		score_R: score.R,
		rally,
		dt,
		W,
		H
	});
	ball.x = r.new_ball_x;
	ball.y = r.new_ball_y;
	ball.vx = r.new_ball_vx;
	ball.vy = r.new_ball_vy;
	paddle.y = r.new_paddle_y;
	rally = r.new_rally;
	const ev = r.events.data;
	return {
		hitPlayer: ev[0] > 0.5,
		hitAi: ev[1] > 0.5,
		wallTop: ev[2] > 0.5,
		wallBottom: ev[3] > 0.5,
		scoredL: ev[4] > 0.5,
		scoredR: ev[5] > 0.5
	};
}

async function onnxCutscene(dt) {
	const r = await workerRun("cutscene", {
		cut: state.cut ? 1.0 : 0.0,
		cutT: state.cutT,
		slowmo: state.slowmo,
		dt
	});
	state.cut = r.new_cut > 0.5;
	state.cutT = r.new_cutT;
	state.slowmo = r.new_slowmo;
	return r.cut_ended > 0.5;
}

async function onnxParticles(dt) {
	const r = await workerRun("particles", {
		particles: { data: particleBuf.slice(), dims: [MAX_P, 7] },
		sparks: { data: sparkBuf.slice(), dims: [MAX_S, 7] },
		dt
	});
	particleBuf.set(r.new_particles.data);
	sparkBuf.set(r.new_sparks.data);
}

/* ═══════════════════════════════════════════════════════════════════
   UI helpers
   ═══════════════════════════════════════════════════════════════════ */

function setPressed(el, on) {
	el.setAttribute("aria-pressed", on ? "true" : "false");
}
function setIntro(on) {
	overlay.classList.toggle("intro", !!on);
}

function showOverlay(title, bodyHtml) {
	setIntro(false);
	overlay.classList.remove("hidden");
	document.getElementById("panelTitle").textContent = title;
	document.getElementById("panelBody").innerHTML = bodyHtml;
}

function hideOverlay() {
	overlay.classList.add("hidden");
}

function setSoundIcon() {
	const i = btnSound.querySelector("i");
	if (i)
		i.className = state.sound
			? "fa-solid fa-volume-high"
			: "fa-solid fa-volume-xmark";
}

function setFXIcon() {
	const i = btnFX.querySelector("i");
	if (i)
		i.className = state.fx
			? "fa-solid fa-wand-magic-sparkles"
			: "fa-solid fa-wand-magic";
}

function setPauseIcon() {
	const i = btnPause.querySelector("i");
	if (i)
		i.className = state.paused ? "fa-solid fa-play" : "fa-solid fa-pause";
}

/* ═══════════════════════════════════════════════════════════════════
   Audio (Web Audio API — stays in JS)
   ═══════════════════════════════════════════════════════════════════ */

let AC = null,
	master = null;
function ensureAudio() {
	if (AC) return;
	AC = new (window.AudioContext || window.webkitAudioContext)();
	master = AC.createGain();
	master.gain.value = 0.12;
	master.connect(AC.destination);
}
function tone(freq, dur = 0.05, type = "sine", gain = 0.2) {
	if (!state.sound) return;
	ensureAudio();
	const t0 = AC.currentTime;
	const o = AC.createOscillator();
	const g = AC.createGain();
	o.type = type;
	o.frequency.setValueAtTime(freq, t0);
	g.gain.setValueAtTime(0.0001, t0);
	g.gain.exponentialRampToValueAtTime(gain, t0 + 0.01);
	g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
	o.connect(g);
	g.connect(master);
	o.start(t0);
	o.stop(t0 + dur + 0.02);
}
function beep(kind) {
	if (!state.sound) return;
	if (kind === "hit") tone(420, 0.025, "sine", 0.14);
	if (kind === "wall") tone(320, 0.03, "sine", 0.12);
	if (kind === "score") {
		tone(180, 0.06, "triangle", 0.16);
		tone(260, 0.06, "triangle", 0.12);
	}
	if (kind === "start") tone(520, 0.05, "sine", 0.12);
	if (kind === "win") {
		tone(330, 0.07, "triangle", 0.16);
		tone(660, 0.08, "sine", 0.12);
	}
}

/* ═══════════════════════════════════════════════════════════════════
   Input (DOM events — stays in JS)
   ═══════════════════════════════════════════════════════════════════ */

let pointerActive = false;

async function setTargetFromClientY(clientY) {
	const rect = canvas.getBoundingClientRect();
	const y = clientY - rect.top;
	paddle.targetY = await clamp(y, paddle.h / 2 + 8, H - paddle.h / 2 - 8);
}

canvas.addEventListener("pointerdown", (e) => {
	pointerActive = true;
	try {
		canvas.setPointerCapture(e.pointerId);
	} catch {}
	setTargetFromClientY(e.clientY);
});
canvas.addEventListener("pointermove", (e) => {
	if (!pointerActive) return;
	setTargetFromClientY(e.clientY);
});
canvas.addEventListener("pointerup", () => {
	pointerActive = false;
});
canvas.addEventListener("mousemove", (e) => {
	if (pointerActive) return;
	setTargetFromClientY(e.clientY);
});
window.addEventListener("touchmove", (e) => e.preventDefault(), {
	passive: false
});

const keys = new Set();
window.addEventListener("keydown", (e) => {
	keys.add(e.key.toLowerCase());
	if (e.key === " ") {
		e.preventDefault();
		if (!state.running) start();
		else togglePause();
	}
	if (e.key.toLowerCase() === "r") hardRestart();
	if (e.key.toLowerCase() === "m") {
		if (state.sound) toggleSound();
	}
});
window.addEventListener("keyup", (e) => keys.delete(e.key.toLowerCase()));

/* ═══════════════════════════════════════════════════════════════════
   Particle buffer management (burst / sparkLine — uses rand, stays JS)
   ═══════════════════════════════════════════════════════════════════ */

function findSlot(buf, max) {
	for (let j = 0; j < max; j++) {
		if (buf[j * 7 + 6] === 0) return j;
	}
	return 0; // overwrite oldest if full
}

function burst(x, y, n) {
	for (let i = 0; i < n; i++) {
		const s = findSlot(particleBuf, MAX_P);
		const off = s * 7;
		particleBuf[off] = x;
		particleBuf[off + 1] = y;
		particleBuf[off + 2] = rand(-240, 240);
		particleBuf[off + 3] = rand(-240, 240);
		particleBuf[off + 4] = rand(0.25, 0.7);
		particleBuf[off + 5] = 0;
		particleBuf[off + 6] = 1;
	}
}

function sparkLine(x, y, vx, vy, n) {
	for (let i = 0; i < n; i++) {
		const s = findSlot(sparkBuf, MAX_S);
		const off = s * 7;
		sparkBuf[off] = x;
		sparkBuf[off + 1] = y;
		sparkBuf[off + 2] = vx * rand(0.2, 0.7) + rand(-60, 60);
		sparkBuf[off + 3] = vy * rand(0.2, 0.7) + rand(-60, 60);
		sparkBuf[off + 4] = rand(0.12, 0.28);
		sparkBuf[off + 5] = 0;
		sparkBuf[off + 6] = 1;
	}
}

/* ═══════════════════════════════════════════════════════════════════
   UI event handlers
   ═══════════════════════════════════════════════════════════════════ */

btnPlay.addEventListener("click", () => start());

if (btnHow) {
	btnHow.addEventListener("click", () => {
		const t = document.getElementById("tiny");
		if (t)
			t.textContent = "HOW: HIT EDGES FOR ANGLE. SPACE PAUSES. R RESTARTS.";
	});
}

btnPause.addEventListener("click", togglePause);
btnSound.addEventListener("click", toggleSound);
btnFX.addEventListener("click", toggleFX);

overlay.addEventListener("click", (e) => {
	if (e.target.closest("button")) return;
	if (!state.running) start();
});

function togglePause() {
	if (!state.running) return;

	state.paused = !state.paused;
	setPressed(btnPause, state.paused);

	if (state.paused) {
		overlayImg.src = "./assets/codepong26.png";
		showOverlay("PAUSED", "PRESS START OR SPACE TO CONTINUE");
	} else {
		overlayImg.src = INTRO_IMG;
		hideOverlay();
	}
}

function toggleSound() {
	state.sound = !state.sound;
	setPressed(btnSound, state.sound);
	setSoundIcon();
	if (state.sound) beep("start");
}

function toggleFX() {
	state.fx = !state.fx;
	setPressed(btnFX, state.fx);
	setFXIcon();
	if (!state.fx) {
		trail.length = 0;
		particleBuf.fill(0);
		sparkBuf.fill(0);
	}
}

/* ═══════════════════════════════════════════════════════════════════
   Match-point cutscene trigger (DOM + condition — stays JS)
   ═══════════════════════════════════════════════════════════════════ */

function maybeMatchPointCutscene() {
	if (!state.fx) return;
	const mp =
		(score.L === score.toWin - 1 && score.R <= score.toWin - 2) ||
		(score.R === score.toWin - 1 && score.L <= score.toWin - 2) ||
		(score.L === score.toWin - 1 && score.R === score.toWin - 1);

	if (!mp) return;

	state.cut = true;
	state.cutT = 0;
	state.slowmo = 0.45;

	cutText.textContent =
		score.L === score.toWin - 1 && score.R === score.toWin - 1
			? "FINAL POINT"
			: "MATCH POINT";
	cutSub.textContent = "SLOW MO";
	cutsceneEl.classList.add("on");
	cutsceneEl.setAttribute("aria-hidden", "false");
}

/* ═══════════════════════════════════════════════════════════════════
   Game lifecycle (state machine + DOM + rand — stays JS)
   ═══════════════════════════════════════════════════════════════════ */

async function resetRound(direction) {
	if (direction === undefined) direction = Math.random() < 0.5 ? -1 : 1;
	ball.x = W / 2;
	ball.y = H / 2;

	const angle = rand(-0.28, 0.28);
	const base = ball.speed;

	ball.vx = Math.cos(angle) * base * direction;
	ball.vy = Math.sin(angle) * base;

	if (state.fx) {
		burst(ball.x, ball.y, 8);
		await onnxShock(3);
	}
	beep("start");

	maybeMatchPointCutscene();
}

async function resetGame() {
	score.L = 0;
	score.R = 0;
	sL.textContent = "0";
	sR.textContent = "0";

	rally = 0;
	ball.speed = 560;
	paddle.h = 110;
	ai.h = 110;

	paddle.y = H / 2;
	paddle.targetY = H / 2;

	ai.y = H / 2;
	ai.memoryY = H / 2;

	trail.length = 0;
	particleBuf.fill(0);
	sparkBuf.fill(0);

	await resetRound();
}

async function start() {
	state.running = true;
	state.paused = false;
	setPressed(btnPause, false);
	setPauseIcon();
	setIntro(false);
	hideOverlay();
	if (playText) playText.textContent = "START";
	await resetGame();
}

const hardRestart = start;

async function checkWinOrReset(nextDir) {
	if (score.L >= score.toWin) {
		endGame(true);
		return;
	}
	if (score.R >= score.toWin) {
		endGame(false);
		return;
	}
	await resetRound(nextDir);
}

function endGame(playerWon) {
	state.running = true;
	state.paused = true;
	setPressed(btnPause, true);
	setPauseIcon();

	overlayImg.src = END_IMG;

	const title = playerWon ? "YOU WIN" : "AI WINS";
	const stats = `FINAL SCORE: ${score.L} - ${score.R}`;
	const msg = playerWon
		? "NICE WORK. RUN IT BACK AND PUSH FOR A PERFECT GAME."
		: "CLOSE. YOU CAN BEAT IT. STAY CALM AND USE THE EDGES OF THE PADDLE.";

	showOverlay(title, `${stats}<br><br>${msg}`);

	if (playText) playText.textContent = "PLAY AGAIN";
	beep("win");
}

/* ═══════════════════════════════════════════════════════════════════
   Rendering (Canvas 2D — stays JS)
   ═══════════════════════════════════════════════════════════════════ */

function roundRect(x, y, w, h, r) {
	const rr = Math.min(r, w / 2, h / 2);
	ctx.beginPath();
	ctx.moveTo(x + rr, y);
	ctx.arcTo(x + w, y, x + w, y + h, rr);
	ctx.arcTo(x + w, y + h, x, y + h, rr);
	ctx.arcTo(x, y + h, x, y, rr);
	ctx.arcTo(x, y, x + w, y, rr);
	ctx.closePath();
}

function draw() {
	ctx.clearRect(0, 0, W, H);

	const sh = state.fx ? state.shake : 0;
	const sx = sh ? rand(-sh, sh) : 0;
	const sy = sh ? rand(-sh, sh) : 0;

	ctx.save();
	ctx.translate(sx, sy);

	const t = state.time;
	const glow = (x, y, a, clr) => {
		const g = ctx.createRadialGradient(x, y, 10, x, y, Math.max(W, H));
		g.addColorStop(0, `rgba(${clr},${a})`);
		g.addColorStop(1, "rgba(0,0,0,0)");
		ctx.fillStyle = g;
		ctx.fillRect(0, 0, W, H);
	};
	glow(W * 0.22, H * 0.28, 0.1 + 0.02 * Math.sin(t * 0.7), "255,79,216");
	glow(
		W * 0.8,
		H * 0.34,
		0.09 + 0.02 * Math.sin(t * 0.8 + 1.2),
		"0,229,255"
	);
	glow(
		W * 0.55,
		H * 0.86,
		0.07 + 0.02 * Math.sin(t * 0.65 + 2.0),
		"124,92,255"
	);

	ctx.globalAlpha = 0.26;
	ctx.fillStyle = "rgba(255,255,255,0.22)";
	for (let y = 12; y < H; y += 22) ctx.fillRect(W / 2 - 1, y, 2, 12);
	ctx.globalAlpha = 1;

	if (state.fx && trail.length > 1) {
		for (let i = 0; i < trail.length - 1; i++) {
			const a = trail[i],
				b = trail[i + 1];
			const age = (state.time - a.t) / 0.2;
			const alpha = (1 - age) * 0.22;
			ctx.strokeStyle = `rgba(0,229,255,${alpha})`;
			ctx.lineWidth = 5 * (1 - age);
			ctx.lineCap = "round";
			ctx.beginPath();
			ctx.moveTo(a.x, a.y);
			ctx.lineTo(b.x, b.y);
			ctx.stroke();
		}
	}

	drawPaddle(paddle.inset, paddle.y, paddle.w, paddle.h, true);
	drawPaddle(W - ai.inset, ai.y, ai.w, ai.h, false);
	drawBall(ball.x, ball.y, ball.r);

	if (state.fx) {
		drawParticles();
	}

	ctx.restore();
}

function drawPaddle(x, y, w, h, isPlayer) {
	const px = x - w / 2;
	const py = y - h / 2;

	const grad = ctx.createLinearGradient(x, py, x, py + h);
	if (isPlayer) {
		grad.addColorStop(0, "rgba(0,229,255,0.95)");
		grad.addColorStop(1, "rgba(124,92,255,0.85)");
	} else {
		grad.addColorStop(0, "rgba(255,79,216,0.85)");
		grad.addColorStop(1, "rgba(0,229,255,0.70)");
	}

	ctx.fillStyle = grad;
	roundRect(px, py, w, h, 10);
	ctx.fill();

	ctx.globalAlpha = 0.3;
	ctx.fillStyle = "rgba(255,255,255,0.34)";
	roundRect(px + 3, py + 6, w - 6, h * 0.18, 10);
	ctx.fill();
	ctx.globalAlpha = 1;
}

function drawBall(x, y, r) {
	ctx.globalAlpha = 0.18;
	ctx.fillStyle = "rgba(0,229,255,1)";
	ctx.beginPath();
	ctx.arc(x, y, r * 2.1, 0, Math.PI * 2);
	ctx.fill();
	ctx.globalAlpha = 1;

	const grad = ctx.createRadialGradient(
		x - r * 0.35,
		y - r * 0.35,
		2,
		x,
		y,
		r * 1.4
	);
	grad.addColorStop(0, "rgba(255,255,255,0.98)");
	grad.addColorStop(0.55, "rgba(0,229,255,0.86)");
	grad.addColorStop(1, "rgba(255,79,216,0.22)");

	ctx.fillStyle = grad;
	ctx.beginPath();
	ctx.arc(x, y, r, 0, Math.PI * 2);
	ctx.fill();
}

function drawParticles() {
	ctx.fillStyle = "rgba(255,255,255,0.75)";
	for (let i = 0; i < MAX_P; i++) {
		const off = i * 7;
		if (particleBuf[off + 6] === 0) continue;
		const k = 1 - particleBuf[off + 5] / particleBuf[off + 4];
		ctx.globalAlpha = 0.7 * k;
		ctx.fillRect(particleBuf[off], particleBuf[off + 1], 2, 2);
	}
	ctx.globalAlpha = 1;

	ctx.strokeStyle = "rgba(255,79,216,0.45)";
	ctx.lineWidth = 2;
	ctx.lineCap = "round";
	for (let i = 0; i < MAX_S; i++) {
		const off = i * 7;
		if (sparkBuf[off + 6] === 0) continue;
		const k = 1 - sparkBuf[off + 5] / sparkBuf[off + 4];
		ctx.globalAlpha = 0.6 * k;
		ctx.beginPath();
		ctx.moveTo(sparkBuf[off], sparkBuf[off + 1]);
		ctx.lineTo(
			sparkBuf[off] - sparkBuf[off + 2] * 0.02,
			sparkBuf[off + 1] - sparkBuf[off + 3] * 0.02
		);
		ctx.stroke();
	}
	ctx.globalAlpha = 1;
}

async function handleScore(side, nextDir) {
	score[side]++;
	(side === "L" ? sL : sR).textContent = score[side];
	rally = 0;
	if (state.fx) {
		burst(W / 2, H / 2, 10);
		await onnxShock(6);
	}
	beep("score");
	await checkWinOrReset(nextDir);
}

/* ═══════════════════════════════════════════════════════════════════
   Game loop (async — uses ONNX)
   ═══════════════════════════════════════════════════════════════════ */

let last = 0;

async function update(ts) {
	requestAnimationFrame(update);
	if (!W || !H) return;

	const now = ts * 0.001;
	let dt = now - last;
	last = now;

	// ── dt clamp ← ONNX ──
	dt = await clamp(dt, 0.008, 0.02);
	dt *= state.slowmo;

	state.time += dt;
	state.shake = Math.max(0, state.shake - dt * 22);

	// ── Cutscene ← ONNX ──
	const cutEnded = await onnxCutscene(dt);
	if (cutEnded) {
		cutsceneEl.classList.remove("on");
		cutsceneEl.setAttribute("aria-hidden", "true");
	}

	// ── Keyboard input (clamp ← ONNX) ──
	let keyDir = 0;
	if (keys.has("w") || keys.has("arrowup")) keyDir -= 1;
	if (keys.has("s") || keys.has("arrowdown")) keyDir += 1;
	if (keyDir) {
		paddle.targetY = await clamp(
			paddle.targetY + keyDir * 780 * dt,
			paddle.h / 2 + 8,
			H - paddle.h / 2 - 8
		);
	}

	if (!state.running) {
		draw();
		return;
	}
	if (state.paused) {
		draw();
		return;
	}

	// ── AI ← ONNX ──
	await onnxAiUpdate(dt);

	// ── Physics + collision ← ONNX ──
	const events = await onnxPhysics(dt);

	// ── Trail (JS — render only) ──
	if (state.fx) {
		trail.push({ x: ball.x, y: ball.y, t: state.time });
		if (trail.length > PERF.maxTrail) trail.shift();
	} else {
		trail.length = 0;
	}

	// ── Process events from physics ──
	if (events.wallTop || events.wallBottom) {
		if (state.fx)
			sparkLine(ball.x, ball.y, ball.vx * 0.03, events.wallTop ? 200 : -200, 6);
		beep("wall");
	}

	if (events.hitPlayer || events.hitAi) {
		if (state.fx) {
			burst(ball.x, ball.y, 6);
			sparkLine(ball.x, ball.y, events.hitPlayer ? 220 : -220, ball.vy * 0.05, 6);
			await onnxShock(4);
		}
		beep("hit");
	}

	if (events.scoredR) await handleScore("R", -1);
	if (events.scoredL) await handleScore("L", 1);

	// ── Particles ← ONNX ──
	if (state.fx) {
		await onnxParticles(dt);
	}

	draw();
}

/* ═══════════════════════════════════════════════════════════════════
   Init — load ONNX, then start
   ═══════════════════════════════════════════════════════════════════ */

async function init() {
	resize();
	paddle.y = H / 2;
	paddle.targetY = H / 2;
	ai.y = H / 2;
	ai.memoryY = H / 2;

	setSoundIcon();
	setFXIcon();
	setPauseIcon();

	overlayImg.src = INTRO_IMG;
	setIntro(true);
	overlay.classList.remove("hidden");
	if (playText) playText.textContent = "LOADING…";

	await workerInit();

	if (playText) playText.textContent = "START";

	requestAnimationFrame(update);
}

init();
