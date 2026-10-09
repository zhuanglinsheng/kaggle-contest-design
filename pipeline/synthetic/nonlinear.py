"""Non-steady-state HJB and structural simulator for PaperJK7, Chapter 4.

Time is measured in days. The legacy simulator is deliberately not overwritten.
The public readout uses the paper's endpoint approximation, with increments of
the SAME signal Brownian motion that drives the contestants' continuous filter.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from scipy.linalg import solve_banded
from scipy.special import ndtr
import math


@dataclass(frozen=True)
class Primitives:
    T: float = 90.0
    theta: float = 1.0
    c_i: float = 1.2
    c_j: float = 1.5
    sigma: float = 2.0
    lamb: float = 1.0
    S0: float = 4.0
    mu0: float = 0.0


def variance(t, p):
    steady = p.sigma / np.sqrt(p.lamb)
    z = np.tanh(p.sigma * np.sqrt(p.lamb) * np.asarray(t))
    return steady * (p.S0 + steady * z) / (steady + p.S0 * z)


@dataclass
class Equilibrium:
    p: Primitives
    t: np.ndarray
    y: np.ndarray
    effort: np.ndarray  # (time, state, player)
    diagnostics: dict

    def at(self, t, y):
        pos = np.clip(t / self.t[-1] * (len(self.t) - 1), 0, len(self.t) - 1)
        i = min(int(pos), len(self.t) - 2)
        f = pos - i
        return np.array([
            (1 - f) * np.interp(y, self.y, self.effort[i, :, k])
            + f * np.interp(y, self.y, self.effort[i + 1, :, k])
            for k in (0, 1)
        ])

    def save(self, path):
        np.savez_compressed(path, t=self.t, y=self.y, effort=self.effort,
                            primitives=np.array(list(asdict(self.p).values())))


def solve_equilibrium(p, *, dy=0.2, dt=1/48, extent=7.0):
    """Crank--Nicolson diffusion with a Heun correction for the HJB nonlinearity.

    Solve normalized values U=V/theta backward, with Gaussian terminal rewards.
    Both policies are obtained from value gradients, not the first-order kernel.
    """
    n = int(np.ceil(p.T / dt))
    t = np.linspace(0, p.T, n + 1)
    dt = p.T / n
    width = np.ceil(extent * np.sqrt(p.S0 + p.sigma**2 * p.T) / dy) * dy
    y = np.arange(-width, width + dy/2, dy)
    S = variance(t, p)
    u = np.column_stack((ndtr(y / np.sqrt(S[-1])), ndtr(-y / np.sqrt(S[-1]))))
    policies = np.empty((n + 1, len(y), 2), dtype=np.float32)
    band = np.zeros((3, len(y) - 2))
    min_policy = 0.0

    def gradient(values):
        return np.gradient(values, dy, axis=0, edge_order=2)

    def nonlinear(values):
        g = gradient(values)
        cross = g[:, 0] * g[:, 1]
        return p.theta * np.column_stack((
            g[:, 0]**2 / (2*p.c_i) + cross / p.c_j,
            g[:, 1]**2 / (2*p.c_j) + cross / p.c_i))

    for k in range(n, -1, -1):
        g = gradient(u)
        effort = g * np.array([p.theta / p.c_i, -p.theta / p.c_j])
        min_policy = min(min_policy, float(effort.min()))
        if min_policy < -1e-7:
            raise RuntimeError(f"Non-monotone value gradient: {min_policy}")
        policies[k] = np.maximum(effort, 0)
        if k == 0:
            break
        a = p.lamb * float(variance((t[k] + t[k-1])/2, p))**2
        r = dt * a / (4 * dy**2)
        band[0, 1:] = -r
        band[1] = 1 + 2*r
        band[2, :-1] = -r
        base = u[1:-1] + r * (u[:-2] - 2*u[1:-1] + u[2:])
        base[0] += r*u[0]
        base[-1] += r*u[-1]
        f = nonlinear(u)
        predictor = u.copy()
        predictor[1:-1] = solve_banded((1, 1), band, base + dt*f[1:-1], check_finite=False)
        u[1:-1] = solve_banded((1, 1), band,
                              base + dt/2*(f[1:-1] + nonlinear(predictor)[1:-1]),
                              check_finite=False)
    diag = dict(dy=dy, dt=dt, extent=extent, nodes_y=len(y), nodes_t=n+1,
                min_raw_policy=min_policy, max_policy=float(policies.max()),
                value_min=float(u.min()), value_max=float(u.max()))
    return Equilibrium(p, t, y, policies, diag)


def simulate(eq, *, seed, ratio=15.0, dt=1/96):
    """Thinning on an event-augmented Euler mesh; exact candidate event times.

    Mesh includes every Poisson candidate time, so readout intervals are elapsed
    event time, never the numerical integration step. All accepted updates are
    retained, including multiple submissions within a regular grid interval.
    """
    p = eq.p
    initial, brownian, poisson, uniform, bridge = [np.random.default_rng(s) for s in
                                                  np.random.SeedSequence(seed).spawn(5)]
    bound = max(1e-12, ratio * float(eq.effort.max()) * 1.001)
    candidates = []
    for player in (0, 1):
        count = poisson.poisson(bound * p.T)
        candidates.extend((float(t), player) for t in poisson.uniform(0, p.T, count))
    candidates.sort()
    regular = np.linspace(0, p.T, int(np.ceil(p.T/dt)) + 1)
    regular_increments = brownian.normal(size=(len(regular)-1,2))*np.sqrt(np.diff(regular))[:,None]
    nodes = np.unique(np.r_[regular, [c[0] for c in candidates]])
    candidate_map = {t: k for t, k in candidates}
    latent = initial.normal(p.mu0, np.sqrt(p.S0))
    mean, signal_b, last_b, last_event = p.mu0, 0.0, 0.0, 0.0
    held = p.mu0
    rows = [(0.0, latent, mean, held, p.S0, *eq.at(0, mean))]
    events = []
    integrated = np.zeros(2)
    max_state = abs(mean)
    regular_index = 0
    remaining_increment = regular_increments[0].copy()
    for t0, t1 in zip(nodes[:-1], nodes[1:]):
        step = t1 - t0
        m = eq.at(t0, mean)
        s = float(variance(t0, p))
        # Brownian bridges retain the SAME regular-grid shock path when r and
        # hence candidate times change during the submission-rate calibration.
        endpoint = regular[regular_index+1]
        if t1 == endpoint:
            increment = remaining_increment.copy()
        else:
            fraction = step/(endpoint-t0)
            increment = fraction*remaining_increment + bridge.normal(size=2)*np.sqrt(step*(1-fraction))
        remaining_increment -= increment
        dw, db = increment
        if t1 == endpoint and regular_index+1 < len(regular_increments):
            regular_index += 1
            remaining_increment = regular_increments[regular_index].copy()
        drift = m[0] - m[1]
        mean += drift*step + p.lamb*s*(latent-mean)*step + np.sqrt(p.lamb)*s*db
        latent += drift*step + p.sigma*dw
        signal_b += db
        integrated += m*step
        max_state = max(max_state, abs(mean))
        if t1 in candidate_map:
            player = candidate_map[t1]
            intensity = ratio * eq.at(t1, mean)[player]
            if intensity > bound * (1 + 1e-12):
                raise RuntimeError("Thinning upper bound violated")
            if uniform.uniform() < intensity / bound:
                h = t1-last_event
                held = latent + (signal_b-last_b)/(np.sqrt(p.lamb)*h)
                events.append((t1, player, held, h))
                last_event, last_b = t1, signal_b
        rows.append((t1, latent, mean, held, float(variance(t1,p)), *eq.at(t1,mean)))
    if max_state >= 0.9 * eq.y[-1]:
        raise RuntimeError("Simulated state too close to HJB boundary")
    return dict(seed=seed, ratio=ratio, T=p.T, events=np.asarray(events).reshape(-1,4),
                path=np.asarray(rows), expected_counts=ratio*integrated,
                initial_y=rows[0][1], thinning_bound=bound)


def likelihood_data(contests, *, estimate_r=False, dt=1/24):
    """Flatten exact event-aligned integration grids for a sequential Stan model."""
    steps, times, held, obs, player = [], [], [], [], []
    starts, ends, horizons, ratios = [], [], [], []
    for c in contests:
        starts.append(len(steps)+1)
        event_map = {float(e[0]): e for e in c['events']}
        grid = np.unique(np.r_[np.linspace(0, c['T'], int(np.ceil(c['T']/dt))+1),
                               c['events'][:,0]])
        previous = 0.0  # initial public readout is the specified common mu_0=0
        for t0,t1 in zip(grid[:-1],grid[1:]):
            e = event_map.get(float(t1))
            steps.append(t1-t0)
            times.append(t0)
            held.append(previous)
            obs.append(0.0 if e is None else float(e[2]))
            player.append(0 if e is None else int(e[1])+1)
            if e is not None:
                previous = float(e[2])
        ends.append(len(steps))
        horizons.append(c['T'])
        ratios.append(c['ratio'])
    # One empirical-Bayes prior, not one prior per replicated contest.
    centers = [float(c['events'][0,2]) for c in contests if len(c['events'])]
    return dict(J=len(contests), N=len(steps), start=starts, end=ends, T=horizons,
                ratio=ratios, dt=steps, t=times, held=held, obs=obs, player=player,
                theta=1.0, estimate_r=int(estimate_r), mu_prior_center=float(np.mean(centers)))


def log_likelihood(data, params):
    """Independent Python likelihood used to audit integration-grid sensitivity."""
    ci,cj,sigma,lamb,S0,mu=[float(params[k]) for k in ('c_i','c_j','sigma','lambda','S0','mu_0')]
    steady=sigma/math.sqrt(lamb)
    t=np.asarray(data['t']); dt=np.asarray(data['dt'])
    z=np.tanh(sigma*math.sqrt(lamb)*t)
    S=steady*(S0+steady*z)/(steady+S0*z)
    ze=np.tanh(sigma*math.sqrt(lamb)*(t+dt))
    Se=steady*(S0+steady*ze)/(steady+S0*ze)
    weights=-np.expm1(-lamb*S*dt)
    total=0.0
    for j in range(data['J']):
        lo,hi=data['start'][j]-1,data['end'][j]
        Q=S[lo:hi]+sigma*sigma*(data['T'][j]-t[lo:hi])
        Qe=Se[lo:hi]+sigma*sigma*(data['T'][j]-t[lo:hi]-dt[lo:hi])
        log_norm=-0.5*np.log(2*math.pi*Q)
        log_norm_end=-0.5*np.log(2*math.pi*Qe)
        r=float(params['r_free']) if data['estimate_r'] else data['ratio'][j]
        perceived,latent_mean,latent_var,elapsed=mu,mu,S0,0.
        for local,n in enumerate(range(lo,hi)):
            kernel=data['theta']*math.exp(log_norm[local]-perceived*perceived/(2*Q[local]))
            mi,mj=kernel/ci,kernel/cj
            gap=mi-mj
            perceived+=weights[n]*(data['held'][n]-perceived+gap/(lamb*S[n]))
            latent_mean+=gap*dt[n]
            latent_var+=sigma*sigma*dt[n]
            elapsed+=dt[n]
            total-=r*(mi+mj)*dt[n]
            if data['player'][n]:
                noise=1/(lamb*elapsed)
                v=latent_var+noise
                innovation=data['obs'][n]-latent_mean
                total+=math.log(r*data['theta']/(ci if data['player'][n]==1 else cj))+log_norm_end[local]-perceived*perceived/(2*Qe[local])
                total+=-.5*(math.log(2*math.pi*v)+innovation*innovation/v)
                latent_mean+=latent_var/v*innovation
                latent_var=latent_var*noise/v
                elapsed=0.
    return total
