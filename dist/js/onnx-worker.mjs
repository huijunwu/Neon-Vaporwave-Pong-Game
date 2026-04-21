/*
 * ONNX inference Worker — all model loading and session.run() happen here,
 * keeping the main thread free for rendering and input.
 */

import * as ort from "../node_modules/onnxruntime-web/dist/ort.all.min.mjs";

ort.env.wasm.wasmPaths = new URL("../node_modules/onnxruntime-web/dist/", import.meta.url).href;
ort.env.wasm.numThreads = 1;

const sessions = {};

function scalar(v) {
	return new ort.Tensor("float32", Float32Array.from([v]), []);
}


const handlers = {
	async init({ base, models }) {
		for (const name of models) {
			sessions[name] = await ort.InferenceSession.create(
				base + "pong_" + name + ".onnx"
			);
		}
		self.postMessage({ type: "ready" });
	},

	async run({ id, model, inputs }) {
		const tensorInputs = {};
		for (const [key, value] of Object.entries(inputs)) {
			if (typeof value === "number") {
				tensorInputs[key] = scalar(value);
			} else if (value && value.data && value.dims) {
				tensorInputs[key] = new ort.Tensor(
					"float32",
					new Float32Array(value.data),
					value.dims
				);
			}
		}

		const result = await sessions[model].run(tensorInputs);

		const outputs = {};
		for (const [key, tensor] of Object.entries(result)) {
			if (tensor.dims.length === 0) {
				outputs[key] = tensor.data[0];
			} else {
				outputs[key] = {
					data: tensor.data.slice(),
					dims: Array.from(tensor.dims)
				};
			}
		}

		self.postMessage({ id, type: "result", outputs });
	}
};

self.onmessage = async (e) => {
	const handler = handlers[e.data.type];
	if (!handler) return;
	try {
		await handler(e.data);
	} catch (err) {
		self.postMessage({ id: e.data.id, type: "error", message: err.message });
	}
};
