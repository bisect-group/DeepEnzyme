"""Lazy AdamW for embedding tables: only the rows a sample reads are updated, and each row catches up the steps it skipped.

torch AdamW updates every row of a table at every step, with a zero gradient for the rows that the sample does not use.
That update is decoupled weight decay, the decay of the moments and the step that the old moment gives. A row that is touched
at step t is first brought from its last step to t - 1 with the same per-step recurrence and g = 0, then step t runs with the
real gradient. The first `CATCH_UP` skipped steps run one by one. The moment m is then below 1e-12 of its start value, so its
step on the weight is below float32 rounding. The remaining steps only decay the weight (exact product of the float32
per-step factors of the learning-rate schedule), m and v.

The learning rate of step j comes from a per-epoch table: all steps of an epoch use the rate of that epoch.
"""

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only environment
    triton = None

CATCH_UP = 256
BETAS = (0.9, 0.999)
EPS = 1e-8


class Schedule:
    """Per-epoch learning rates of the StepLR(3, 0.9) schedule of the trainer, and the weight-decay prefix sums."""

    def __init__(self, lr, weight_decay, epochs, steps_per_epoch, device):
        param = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.AdamW([param], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
        optimizer.step()  # no gradients: only tells the scheduler that the optimizer ran
        rates = []
        for epoch in range(1, epochs + 2):
            rates.append(optimizer.param_groups[0]["lr"])
            if epoch // 10 > 0 and epoch % 10 > 0:
                scheduler.step()
        self.steps_per_epoch = steps_per_epoch
        lr32 = torch.tensor(rates, dtype=torch.float32)
        decay32 = 1 - lr32 * torch.tensor(weight_decay, dtype=torch.float32)  # the factor of one step, rounded to float32 as in the optimizer
        self.lr = lr32.to(device)
        self.log_decay = decay32.double().log().to(device)
        self.prefix = torch.cat((torch.zeros(1, dtype=torch.float64), (self.log_decay.cpu() * steps_per_epoch).cumsum(0))).to(device)

    def log_decay_until(self, steps):
        """Sum of the log decay factors of steps 1..steps (int tensor)."""
        epoch = steps // self.steps_per_epoch
        return self.prefix[epoch] + (steps - epoch * self.steps_per_epoch).double() * self.log_decay[epoch]


def adamw_step(p, m, v, g, lr, step, wd, b1=BETAS[0], b2=BETAS[1], eps=EPS):
    """One AdamW step in torch AdamW's order. `step` is an int tensor, all other tensors are float32 and broadcast against p."""
    p = p * (1 - lr * wd)
    m = m + (g - m) * (1 - b1)
    v = v * b2 + (1 - b2) * g * g
    bias1 = (1 - torch.exp(step.double() * math.log(b1))).float()
    bias2 = (1 - torch.exp(step.double() * math.log(b2))).float().sqrt()
    return p - lr / bias1 * m / (v.sqrt() / bias2 + eps), m, v


if triton is not None:

    @triton.jit
    def _step(p, m, v, g, lr, j, wd, b1, b2, eps, lb1, lb2):
        p = p * (1.0 - lr * wd)
        m = m + (g - m) * (1.0 - b1)
        v = v * b2 + (1.0 - b2) * g * g
        bias1 = (1.0 - tl.exp(j.to(tl.float64) * lb1)).to(tl.float32)
        bias2 = tl.sqrt_rn((1.0 - tl.exp(j.to(tl.float64) * lb2)).to(tl.float32))
        denom = tl.div_rn(tl.sqrt_rn(v), bias2) + eps
        return p - tl.div_rn(tl.div_rn(lr, bias1) * m, denom), m, v

    @triton.jit
    def _log_decay(x, prefix_ptr, log_decay_ptr, n_steps):
        epoch = x // n_steps
        return tl.load(prefix_ptr + epoch) + (x - epoch * n_steps).to(tl.float64) * tl.load(log_decay_ptr + epoch)

    @triton.jit(do_not_specialize=["target"])
    def _advance_kernel(p_ptr, m_ptr, v_ptr, rows_ptr, last_ptr, target, lr_ptr, log_decay_ptr, prefix_ptr, n_steps, dim, wd, b1, b2, eps, lb1, lb2, catch_up: tl.constexpr, block: tl.constexpr):
        row = tl.load(rows_ptr + tl.program_id(0)).to(tl.int64)
        last = tl.load(last_ptr + row)
        skipped = target - last
        n_run = tl.minimum(skipped, catch_up)
        offsets = tl.arange(0, block)
        mask = offsets < dim
        at = row * dim + offsets
        p = tl.load(p_ptr + at, mask=mask, other=0.0)
        m = tl.load(m_ptr + at, mask=mask, other=0.0)
        v = tl.load(v_ptr + at, mask=mask, other=0.0)
        zero = tl.zeros([block], tl.float32)
        for i in range(0, n_run):
            j = last + 1 + i
            p, m, v = _step(p, m, v, zero, tl.load(lr_ptr + (j - 1) // n_steps), j, wd, b1, b2, eps, lb1, lb2)
        rest = skipped - n_run
        if rest > 0:
            span = _log_decay(target, prefix_ptr, log_decay_ptr, n_steps) - _log_decay(last + n_run, prefix_ptr, log_decay_ptr, n_steps)
            p = p * tl.exp(span).to(tl.float32)
            m = m * tl.exp(rest.to(tl.float64) * lb1).to(tl.float32)
            v = v * tl.exp(rest.to(tl.float64) * lb2).to(tl.float32)
        tl.store(p_ptr + at, p, mask=mask)
        tl.store(m_ptr + at, m, mask=mask)
        tl.store(v_ptr + at, v, mask=mask)
        tl.store(last_ptr + row, target)

    @triton.jit(do_not_specialize=["step"])
    def _step_kernel(p_ptr, m_ptr, v_ptr, g_ptr, rows_ptr, last_ptr, step, lr_ptr, n_steps, dim, wd, b1, b2, eps, lb1, lb2, block: tl.constexpr):
        pid = tl.program_id(0)
        row = tl.load(rows_ptr + pid).to(tl.int64)
        offsets = tl.arange(0, block)
        mask = offsets < dim
        at = row * dim + offsets
        p = tl.load(p_ptr + at, mask=mask, other=0.0)
        m = tl.load(m_ptr + at, mask=mask, other=0.0)
        v = tl.load(v_ptr + at, mask=mask, other=0.0)
        g = tl.load(g_ptr + pid * dim + offsets, mask=mask, other=0.0)
        p, m, v = _step(p, m, v, g, tl.load(lr_ptr + (step - 1) // n_steps), step, wd, b1, b2, eps, lb1, lb2)
        tl.store(p_ptr + at, p, mask=mask)
        tl.store(m_ptr + at, m, mask=mask)
        tl.store(v_ptr + at, v, mask=mask)
        tl.store(last_ptr + row, step)


class LazyRows:
    """Lazy AdamW state of embedding tables that are read with the same row indices. `weights` are updated in place."""

    def __init__(self, weights, schedule, weight_decay, use_kernel=None):
        self.weights = {name: w.data for name, w in weights.items()}
        self.schedule, self.weight_decay = schedule, weight_decay
        self.exp_avg = {name: torch.zeros_like(w) for name, w in self.weights.items()}
        self.exp_avg_sq = {name: torch.zeros_like(w) for name, w in self.weights.items()}
        first = next(iter(self.weights.values()))
        self.last = torch.zeros(first.shape[0], dtype=torch.int64, device=first.device)
        self.use_kernel = first.is_cuda and triton is not None if use_kernel is None else use_kernel

    def _scalars(self):
        return self.weight_decay, BETAS[0], BETAS[1], EPS, math.log(BETAS[0]), math.log(BETAS[1])

    def advance(self, rows, target):
        """Bring the rows (unique indices) from their last step to `target` with g = 0."""
        for name, weight in self.weights.items():
            if self.use_kernel:
                s = self.schedule
                dim = weight.shape[1]
                _advance_kernel[(len(rows),)](weight, self.exp_avg[name], self.exp_avg_sq[name], rows, self.last, target, s.lr, s.log_decay, s.prefix, s.steps_per_epoch, dim, *self._scalars(), catch_up=CATCH_UP, block=triton.next_power_of_2(dim))
            else:
                self._advance_reference(name, rows, target)
        if self.use_kernel:
            return
        self.last[rows] = target

    def _advance_reference(self, name, rows, target):
        s = self.schedule
        last = self.last[rows]
        skipped = target - last
        n_run = skipped.clamp(max=CATCH_UP)
        p, m, v = (t[rows] for t in (self.weights[name], self.exp_avg[name], self.exp_avg_sq[name]))
        zero = torch.zeros_like(p)
        for i in range(int(n_run.max()) if len(rows) else 0):
            j = last + 1 + i
            new = adamw_step(p, m, v, zero, s.lr[((j - 1) // s.steps_per_epoch).clamp(max=len(s.lr) - 1)][:, None], j[:, None], self.weight_decay)
            p, m, v = (torch.where((i < n_run)[:, None], a, b) for a, b in zip(new, (p, m, v)))
        rest = skipped - n_run
        span = s.log_decay_until(torch.full_like(last, target)) - s.log_decay_until(last + n_run)
        p = p * span.exp().float()[:, None]
        m = m * (rest.double() * math.log(BETAS[0])).exp().float()[:, None]
        v = v * (rest.double() * math.log(BETAS[1])).exp().float()[:, None]
        for t, new in zip((self.weights[name], self.exp_avg[name], self.exp_avg_sq[name]), (p, m, v)):
            t[rows] = new

    def step(self, rows, grads, step):
        """AdamW step `step` for the rows (already at step - 1) with the gradient rows `grads[name]` of shape [len(rows), dim]."""
        for name, weight in self.weights.items():
            grad = grads[name].contiguous()
            if self.use_kernel:
                s = self.schedule
                dim = weight.shape[1]
                _step_kernel[(len(rows),)](weight, self.exp_avg[name], self.exp_avg_sq[name], grad, rows, self.last, step, s.lr, s.steps_per_epoch, dim, *self._scalars(), block=triton.next_power_of_2(dim))
            else:
                lr = self.schedule.lr[(step - 1) // self.schedule.steps_per_epoch]
                new = adamw_step(weight[rows], self.exp_avg[name][rows], self.exp_avg_sq[name][rows], grad, lr, torch.tensor(step), self.weight_decay)
                for t, value in zip((weight, self.exp_avg[name], self.exp_avg_sq[name]), new):
                    t[rows] = value
        if not self.use_kernel:
            self.last[rows] = step

    def finalize(self, step):
        """Catch every row up to `step`, so the state equals the state of the dense optimizer."""
        self.advance(torch.arange(len(self.last), device=self.last.device), step)

    def state(self, step):
        """Optimizer state entries in torch AdamW format. Call after `finalize`."""
        return {name: {"step": torch.tensor(float(step)), "exp_avg": self.exp_avg[name].cpu(), "exp_avg_sq": self.exp_avg_sq[name].cpu()} for name in self.weights}

    def load(self, entries, step):
        for name, entry in entries.items():
            self.exp_avg[name].copy_(entry["exp_avg"])
            self.exp_avg_sq[name].copy_(entry["exp_avg_sq"])
        self.last.fill_(step)


class LazyEmbedding(torch.nn.Module):
    """Embedding that reads the unique rows of the sample and routes the gradient of those rows only.

    `bind(rows, inverse)` is called before each forward with the unique word ids of the sample and the inverse index.
    The weight has no gradient; `rows_grad` after backward is the gradient of the unique rows.
    """

    def __init__(self, embedding):
        super().__init__()
        self.weight = embedding.weight
        self.weight.requires_grad_(False)

    def bind(self, rows, inverse):
        self.rows, self.inverse, self.vectors = rows, inverse, None

    def forward(self, words):
        self.vectors = self.weight.index_select(0, self.rows).requires_grad_(torch.is_grad_enabled())
        return torch.nn.functional.embedding(self.inverse, self.vectors).reshape(*words.shape, -1)

    @property
    def rows_grad(self):
        return self.vectors.grad
