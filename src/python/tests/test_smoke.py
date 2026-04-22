"""Smoke test: verify PongStepModule + PongPolicyModule work with torch.compile and torch.vmap."""

import torch
from pong.onnx_modules import PongStepModule, PongPolicyModule, PongState


def test_basic_rollout():
    env = PongStepModule()
    seed = torch.tensor(42, dtype=torch.int64)
    state, ts = env.reset(seed)

    for _ in range(200):
        actions = torch.tensor([1, 2])
        state, ts = env.step(state, actions)

    print(f"[basic] 200 steps, score: {state.score_left.item():.0f}-{state.score_right.item():.0f}")


def test_multi_agent():
    env = PongStepModule()
    left = PongPolicyModule(reaction=0.16)
    right = PongPolicyModule(reaction=0.20)

    seed = torch.tensor(7, dtype=torch.int64)
    state, ts = env.reset(seed)
    ls = left.initial_state(torch.tensor(1, dtype=torch.int64))
    rs = right.initial_state(torch.tensor(2, dtype=torch.int64))

    for _ in range(5000):
        la, ls = left.act(ts.obs[0], ls)
        ra, rs = right.act(ts.obs[1], rs)
        state, ts = env.step(state, torch.stack([la, ra]))

    print(f"[multi-agent] 5000 steps, score: {state.score_left.item():.0f}-{state.score_right.item():.0f}")


def test_vmap_batch():
    env = PongStepModule()
    batch_size = 8

    batched_reset = torch.vmap(lambda s: env.reset(s))
    batched_step = torch.vmap(lambda s, a: env.step(s, a))

    seeds = torch.arange(batch_size, dtype=torch.int64)
    states, ts = batched_reset(seeds)

    assert ts.obs.shape == (batch_size, 2, 6), f"Expected (8,2,6), got {ts.obs.shape}"

    for _ in range(20):
        actions = torch.randint(0, 3, (batch_size, 2))
        states, ts = batched_step(states, actions)

    print(f"[vmap] batch={batch_size}, 20 steps, ball_x range: "
          f"[{states.ball_x.min().item():.1f}, {states.ball_x.max().item():.1f}]")


def test_compile():
    env = PongStepModule()

    @torch.compile(fullgraph=True, backend="aot_eager")
    def compiled_step(state: PongState, actions: torch.Tensor):
        return env.step(state, actions)

    seed = torch.tensor(7, dtype=torch.int64)
    state, ts = env.reset(seed)

    for i in range(50):
        actions = torch.tensor([i % 3, (i + 1) % 3])
        state, ts = compiled_step(state, actions)

    print(f"[compile] 50 steps, score: {state.score_left.item():.0f}-{state.score_right.item():.0f}")


def test_compile_plus_vmap():
    env = PongStepModule()
    batch_size = 16

    batched_step = torch.vmap(lambda s, a: env.step(s, a))

    @torch.compile(fullgraph=True, backend="aot_eager")
    def compiled_batched_step(states, actions):
        return batched_step(states, actions)

    seeds = torch.arange(batch_size, dtype=torch.int64)
    states, ts = torch.vmap(lambda s: env.reset(s))(seeds)

    for _ in range(100):
        actions = torch.randint(0, 3, (batch_size, 2))
        states, ts = compiled_batched_step(states, actions)

    print(f"[compile+vmap] batch={batch_size}, 100 steps, "
          f"scores: L={states.score_left.sum().item():.0f} R={states.score_right.sum().item():.0f}")


if __name__ == "__main__":
    tests = [
        ("Basic rollout", test_basic_rollout),
        ("Multi-agent (2 PongPolicyModules)", test_multi_agent),
        ("vmap batched environments", test_vmap_batch),
        ("torch.compile", test_compile),
        ("compile + vmap combined", test_compile_plus_vmap),
    ]

    for i, (name, fn) in enumerate(tests, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {name}")
        print(f"{'='*60}")
        fn()

    print(f"\n\nALL {len(tests)} TESTS PASSED")
