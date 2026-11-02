//! Owned per-call search state. Geometry and scoring never share mutable buffers.
use crate::sweep;

const INF: f32 = f32::INFINITY;
type Point = [f32; 2];
fn norm(v: Point) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}
fn sub(a: Point, b: Point) -> Point {
    [a[0] - b[0], a[1] - b[1]]
}
fn point(p: Point, v: Point, t: f32) -> Point {
    [p[0] + v[0] * t, p[1] + v[1] * t]
}
fn clip(p: Point, low: Point, high: Point) -> Point {
    [p[0].clamp(low[0], high[0]), p[1].clamp(low[1], high[1])]
}
fn dot(a: Point, b: Point) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

#[derive(Clone)]
struct Circle {
    p: Point,
    v: Point,
    r: f32,
    u: f32,
    known: bool,
}
#[derive(Clone)]
struct Body {
    p: Point,
    v: Point,
    e: Point,
    u: f32,
    angle: f32,
    known: bool,
}
#[derive(Clone)]
struct Scene {
    circles: Vec<Circle>,
    bodies: Vec<Body>,
    radius: f32,
    low: Point,
    high: Point,
}
#[derive(Clone, Copy)]
struct Segment {
    end: Point,
    clear: f32,
    hit: f32,
    effort: f32,
}
impl Scene {
    fn run(
        &self,
        starts: &[Point],
        velocities: &[Point],
        durations: &[f32],
        ticks: &[f32],
    ) -> Vec<Segment> {
        let n = starts.len();
        let m = self.circles.len();
        let k = self.bodies.len();
        let mut data = Vec::with_capacity(6 * n + 4 + 6 * m + 8 * k);
        for p in starts {
            data.extend_from_slice(p)
        }
        for v in velocities {
            data.extend_from_slice(v)
        }
        data.extend_from_slice(durations);
        data.extend_from_slice(ticks);
        data.extend_from_slice(&self.low);
        data.extend_from_slice(&self.high);
        for b in &self.circles {
            data.extend_from_slice(&[b.p[0], b.p[1], b.v[0], b.v[1], b.r, b.u])
        }
        for b in &self.bodies {
            data.extend_from_slice(&[b.p[0], b.p[1], b.v[0], b.v[1], b.e[0], b.e[1], b.u, b.angle])
        }
        sweep(&data, n, m, k, self.radius)
            .chunks_exact(5)
            .map(|r| Segment {
                end: [r[0], r[1]],
                clear: r[2],
                hit: r[3],
                effort: r[4],
            })
            .collect()
    }
    fn one(&self, start: Point, velocity: Point, duration: f32, tick: f32) -> Segment {
        self.run(&[start], &[velocity], &[duration], &[tick])[0]
    }
    fn confirmed(&self) -> Self {
        Self {
            circles: self
                .circles
                .iter()
                .filter(|b| b.known)
                .map(|b| Circle { u: 0., ..b.clone() })
                .collect(),
            bodies: self
                .bodies
                .iter()
                .filter(|b| b.known)
                .map(|b| Body { u: 0., ..b.clone() })
                .collect(),
            ..self.clone()
        }
    }
    fn reachable(&self, start: Point, speed: f32, end: f32) -> Self {
        let near = |p: Point, v: Point, r: f32, u: f32| {
            let relative = sub(p, start);
            let t = (-dot(relative, v) / dot(v, v).max(1e-12)).clamp(0., end);
            norm(point(relative, v, t)) <= speed * end + r + 2. + u * end
        };
        Self {
            circles: self
                .circles
                .iter()
                .filter(|b| near(b.p, b.v, b.r, b.u))
                .cloned()
                .collect(),
            bodies: self
                .bodies
                .iter()
                .filter(|b| near(b.p, b.v, norm(b.e) + self.radius, b.u))
                .cloned()
                .collect(),
            ..self.clone()
        }
    }
}

struct Probe {
    free: [bool; 18],
    hit: [f32; 18],
    action_hit: [f32; 18],
    clear: [f32; 18],
}
fn probe(
    scene: &Scene,
    start: Point,
    pending: Point,
    delays: &[f32],
    hold: f32,
    speed: f32,
    velocities: &[Point; 18],
) -> Probe {
    let scene = scene.reachable(
        start,
        speed.max(norm(pending)),
        delays[delays.len() - 1] + hold,
    );
    let mut result = Probe {
        free: [true; 18],
        hit: [INF; 18],
        action_hit: [INF; 18],
        clear: [INF; 18],
    };
    if scene.circles.is_empty() && scene.bodies.is_empty() {
        return result;
    }
    let prefix = scene.run(
        &vec![start; delays.len()],
        &vec![pending; delays.len()],
        delays,
        &vec![0.; delays.len()],
    );
    let mut starts = Vec::with_capacity(delays.len() * 18);
    let mut moves = Vec::with_capacity(starts.capacity());
    let mut ticks = Vec::with_capacity(starts.capacity());
    for (i, d) in delays.iter().enumerate() {
        for v in velocities {
            starts.push(prefix[i].end);
            moves.push(*v);
            ticks.push(*d)
        }
    }
    let segments = scene.run(&starts, &moves, &vec![hold; starts.len()], &ticks);
    for (i, s) in segments.iter().enumerate() {
        let a = i % 18;
        let hit = prefix[i / 18].hit.min(s.hit);
        result.free[a] &= hit.is_infinite();
        result.hit[a] = result.hit[a].min(hit);
        result.action_hit[a] = result.action_hit[a].min(s.hit);
        result.clear[a] = result.clear[a].min(s.clear);
    }
    result
}

#[derive(Clone)]
struct Item {
    p: [f64; 2],
    v: [f64; 2],
    weight: f64,
}
#[derive(Clone)]
struct Node {
    root: usize,
    last: usize,
    p: Point,
    path: Vec<Point>,
    minimum: f32,
    collision: f32,
    action_minimum: f32,
    action_collision: f32,
    edge: f64,
    corner: f64,
    wall: f64,
    aim: f64,
    goal: f64,
    movement: f64,
    score: f64,
}
fn heading(v: Point, directions: &[Point; 18]) -> usize {
    if norm(v) < 1e-4 {
        return 0;
    }
    let mut best = 1;
    for a in 2..9 {
        if dot(v, directions[a]) > dot(v, directions[best]) {
            best = a
        }
    }
    best
}
fn branches(
    node: &Node,
    count: usize,
    tick: f32,
    scene: &Scene,
    items: &[Item],
    target: Option<[f64; 3]>,
    directions: &[Point; 18],
) -> Vec<usize> {
    if count == 18 {
        return (0..18).collect();
    }
    let mut goal = [
        (scene.low[0] + scene.high[0]) * 0.5 - node.p[0],
        (scene.low[1] + scene.high[1]) * 0.5 - node.p[1],
    ];
    if !items.is_empty() {
        let mut best = f64::NEG_INFINITY;
        for item in items {
            let rel = [
                item.p[0] + item.v[0] * tick as f64 - node.p[0] as f64,
                item.p[1] + item.v[1] * tick as f64 - node.p[1] as f64,
            ];
            let score = item.weight * (-(rel[0] * rel[0] + rel[1] * rel[1]).sqrt() / 64.).exp();
            if score > best {
                best = score;
                goal = [rel[0] as f32, rel[1] as f32]
            }
        }
    } else if let Some(t) = target {
        goal = [(t[0] + t[1] * tick as f64 - node.p[0] as f64) as f32, 0.]
    }
    let mut escape = goal;
    let mut closest = INF;
    for b in &scene.circles {
        let r = sub(point(b.p, b.v, tick), node.p);
        let clear = norm(r) - b.r;
        if clear < closest {
            closest = clear;
            escape = if clear < 32. { [-r[0], -r[1]] } else { goal }
        }
    }
    for b in &scene.bodies {
        let r = sub(point(b.p, b.v, tick), node.p);
        let clear = norm(r) - norm(b.e) - scene.radius;
        if clear < closest {
            closest = clear;
            escape = if clear < 32. { [-r[0], -r[1]] } else { goal }
        }
    }
    let ring = [4, 8, 2, 7, 3, 5, 1, 6];
    let last = node.last;
    let focus = last / 9 * 9;
    let around = ring.iter().position(|v| *v == last % 9).unwrap_or(0);
    let choices = [
        last,
        focus + heading(escape, directions),
        focus,
        focus + heading(goal, directions),
        (last + 9) % 18,
        focus + ring[(around + 7) % 8],
        focus + ring[(around + 1) % 8],
        focus + ring[(around + 4) % 8],
    ];
    let mut seen = [false; 18];
    let mut out = Vec::with_capacity(count);
    for a in choices.into_iter().chain(0..18) {
        if !seen[a] {
            seen[a] = true;
            out.push(a);
            if out.len() == count {
                break;
            }
        }
    }
    out
}
fn qualities(
    start: Point,
    end: Point,
    duration: f32,
    tick: f32,
    items: &[Item],
    target: Option<[f64; 3]>,
    bounds: [f64; 2],
) -> (f32, f32) {
    let mut quality = 0_f64;
    for item in items {
        let initial = [
            item.p[0] + item.v[0] * tick as f64,
            item.p[1] + item.v[1] * tick as f64,
        ];
        let r = [initial[0] - start[0] as f64, initial[1] - start[1] as f64];
        let v = [
            item.v[0] - (end[0] - start[0]) as f64 / duration as f64,
            item.v[1] - (end[1] - start[1]) as f64 / duration as f64,
        ];
        let t = (-(r[0] * v[0] + r[1] * v[1]) / (v[0] * v[0] + v[1] * v[1]).max(1e-12))
            .clamp(0., duration as f64);
        let x = initial[0] + item.v[0] * t;
        let y = initial[1] + item.v[1] * t;
        if x >= 0. && x <= bounds[0] && y >= 0. && y <= bounds[1] {
            let dx = r[0] + v[0] * t;
            let dy = r[1] + v[1] * t;
            quality = quality.max(
                item.weight * (-(dx * dx + dy * dy).sqrt() / 64. - 0.04 * (tick as f64 + t)).exp(),
            )
        }
    }
    let aim = target.map_or(0., |t| {
        t[2] * (-0.5 * ((end[0] as f64 - (t[0] + t[1] * (tick + duration) as f64)) / 48.).powi(2))
            .exp()
    });
    (quality as f32, aim as f32)
}
fn argmax(scores: &[f64; 18], allowed: &[bool; 18]) -> usize {
    let mut best = None;
    for i in 0..18 {
        if allowed[i] && best.is_none_or(|b| scores[i] > scores[b]) {
            best = Some(i)
        }
    }
    best.unwrap_or(0)
}

/// Buffer schema: start/pending/low/high, 18 velocities, 18 unit directions,
/// control[26], bounds maxima, target-present/x/v/weight, circles[7], bodies[9],
/// items[5]. Geometry is f32; goals and score accumulation remain f64.
pub fn plan(
    data: &[f64],
    m: usize,
    k: usize,
    q: usize,
    delay: f32,
    late: f32,
    horizon: f32,
    durations: &[f32],
    delays: &[f32],
    complete: bool,
    beam: usize,
    branch_count: usize,
    fast: f32,
    age: f64,
) -> (Vec<f32>, Vec<f64>) {
    let p = |offset| [data[offset] as f32, data[offset + 1] as f32];
    let start = p(0);
    let pending = p(2);
    let low = p(4);
    let high = p(6);
    let velocities: [Point; 18] = std::array::from_fn(|i| p(8 + 2 * i));
    let directions: [Point; 18] = std::array::from_fn(|i| p(44 + 2 * i));
    let control = data[80];
    let bounds = [data[81], data[82]];
    let target = if data[83] > 0.5 {
        Some([data[84], data[85], data[86]])
    } else {
        None
    };
    let mut offset = 87;
    let circles = data[offset..offset + 7 * m]
        .chunks_exact(7)
        .map(|r| Circle {
            p: [r[0] as f32, r[1] as f32],
            v: [r[2] as f32, r[3] as f32],
            r: r[4] as f32,
            u: r[5] as f32,
            known: r[6] > 0.5,
        })
        .collect();
    offset += 7 * m;
    let bodies = data[offset..offset + 9 * k]
        .chunks_exact(9)
        .map(|r| Body {
            p: [r[0] as f32, r[1] as f32],
            v: [r[2] as f32, r[3] as f32],
            e: [r[4] as f32, r[5] as f32],
            u: r[6] as f32,
            angle: r[7] as f32,
            known: r[8] > 0.5,
        })
        .collect();
    offset += 9 * k;
    let items: Vec<Item> = data[offset..offset + 5 * q]
        .chunks_exact(5)
        .map(|r| Item {
            p: [r[0], r[1]],
            v: [r[2], r[3]],
            weight: r[4],
        })
        .collect();
    offset += 5 * q;
    let scene = Scene {
        circles,
        bodies,
        radius: data[offset] as f32,
        low,
        high,
    };
    let confirmed = scene.confirmed();
    let prefix = scene.one(start, pending, delay, 0.);
    let hold = late + 2.;
    let timing = probe(&scene, start, pending, delays, hold, fast, &velocities);
    let mut nodes: Vec<Node> = (0..18)
        .map(|root| Node {
            root,
            last: root,
            p: prefix.end,
            path: vec![start, prefix.end],
            minimum: prefix.clear,
            collision: prefix.hit,
            action_minimum: INF,
            action_collision: INF,
            edge: 0.,
            corner: 0.,
            wall: 0.,
            aim: 0.,
            goal: 0.,
            movement: 0.,
            score: 0.,
        })
        .collect();
    let mut root_effort = [0_f32; 18];
    let mut root_blocked = [false; 18];
    let mut tick = delay;
    let mut checked = 0;
    for (depth, duration) in durations.iter().copied().enumerate() {
        let mut parents = Vec::new();
        let mut actions = Vec::new();
        for (i, node) in nodes.iter().enumerate() {
            let options = if depth == 0 {
                vec![node.root]
            } else {
                branches(
                    node,
                    branch_count,
                    tick,
                    &scene,
                    &items,
                    target,
                    &directions,
                )
            };
            for a in options {
                parents.push(i);
                actions.push(a)
            }
        }
        let starts: Vec<Point> = parents.iter().map(|i| nodes[*i].p).collect();
        let moves: Vec<Point> = actions.iter().map(|a| velocities[*a]).collect();
        let segments = scene.run(
            &starts,
            &moves,
            &vec![duration; starts.len()],
            &vec![tick; starts.len()],
        );
        checked += segments.len();
        let mut next: Vec<Node> = Vec::with_capacity(segments.len());
        for (i, s) in segments.iter().enumerate() {
            let parent = &nodes[parents[i]];
            let root = parent.root;
            if depth == 0 {
                root_effort[root] = s.effort;
                root_blocked[root] =
                    norm(sub(s.end, parent.p)) < 0.1 && norm(velocities[root]) > 0.1
            }
            let mut node = parent.clone();
            node.p = s.end;
            node.last = actions[i];
            node.path.push(s.end);
            node.minimum = node.minimum.min(s.clear);
            node.collision = node.collision.min(s.hit);
            node.action_minimum = node.action_minimum.min(s.clear);
            node.action_collision = node.action_collision.min(s.hit);
            let midpoint = [
                (parent.p[0] + s.end[0]) * 0.5,
                (parent.p[1] + s.end[1]) * 0.5,
            ];
            let distances = [
                (midpoint[0] - low[0]).min(high[0] - midpoint[0]),
                (midpoint[1] - low[1]).min(high[1] - midpoint[1]),
            ];
            let exposure = distances.map(|d| (-0.5 * (d / 32.).powi(2)).exp());
            let (goal, aim) = qualities(parent.p, s.end, duration, tick, &items, target, bounds);
            let weight = duration / horizon;
            node.edge += (exposure[0].max(exposure[1]) * weight) as f64;
            node.corner += (exposure[0] * exposure[1] * weight) as f64;
            node.wall += (s.effort * weight) as f64;
            node.aim += (aim * weight) as f64;
            node.goal = node.goal.max(goal as f64);
            node.movement += (norm(sub(s.end, parent.p)) / (fast * horizon)) as f64;
            node.score = 1.4 * node.goal + 0.5 * node.aim
                - 1.2 * node.edge
                - 0.6 * (1. + control) * node.corner
                - 1.5 * node.wall
                - 0.03 * node.movement
                - (2. * root_effort[root]) as f64
                - (0.5 * ((6. - node.minimum) / 6.).clamp(0., 2.)) as f64;
            next.push(node);
        }
        let width = if depth + 1 == durations.len() {
            1
        } else {
            beam
        };
        let per_root = next.len() / 18;
        nodes = Vec::with_capacity(18 * width);
        for root in 0..18 {
            let mut indices: Vec<usize> = (root * per_root..(root + 1) * per_root).collect();
            indices.sort_by(|a, b| {
                let a = &next[*a];
                let b = &next[*b];
                b.collision
                    .min(horizon + 1.)
                    .partial_cmp(&a.collision.min(horizon + 1.))
                    .unwrap()
                    .then_with(|| {
                        b.action_collision
                            .min(horizon + 1.)
                            .partial_cmp(&a.action_collision.min(horizon + 1.))
                            .unwrap()
                    })
                    .then_with(|| b.score.partial_cmp(&a.score).unwrap())
            });
            for index in indices.into_iter().take(width) {
                nodes.push(next[index].clone())
            }
        }
        tick += duration;
    }
    let safe: [bool; 18] =
        std::array::from_fn(|i| nodes[i].collision.is_infinite() && timing.free[i] && complete);
    let mut allowed = std::array::from_fn(|i| safe[i] && !root_blocked[i]);
    let fallback = !allowed.iter().any(|v| *v);
    let nominal = probe(&confirmed, start, pending, delays, 8., fast, &velocities);
    let known_prefix = confirmed.one(start, pending, late, 0.);
    let mut immediate = [true; 18];
    let mut body_probe = false;
    let mut short_priority = false;
    if fallback {
        if !scene.bodies.is_empty() {
            body_probe = true;
            let mut bodies = scene.clone();
            bodies.circles.clear();
            for b in &mut bodies.bodies {
                b.u = 0.25
            }
            immediate = probe(&bodies, start, pending, delays, hold, fast, &velocities).free;
        }
        let mut eligible = if scene.bodies.is_empty() {
            [true; 18]
        } else {
            std::array::from_fn(|i| immediate[i] && !root_blocked[i])
        };
        if !eligible.iter().any(|v| *v) {
            eligible = [true; 18]
        }
        let combined: [f32; 18] =
            std::array::from_fn(|i| nodes[i].collision.min(timing.hit[i]).min(horizon + 1.));
        let latest = (0..18)
            .filter(|i| eligible[*i])
            .map(|i| combined[i])
            .fold(f32::NEG_INFINITY, f32::max);
        allowed = std::array::from_fn(|i| eligible[i] && combined[i] >= latest - 0.5);
        let recovery: [f32; 18] = std::array::from_fn(|i| {
            nodes[i]
                .action_collision
                .min(timing.action_hit[i])
                .min(horizon + 1.)
        });
        let latest = (0..18)
            .filter(|i| allowed[*i])
            .map(|i| recovery[i])
            .fold(f32::NEG_INFINITY, f32::max);
        for i in 0..18 {
            allowed[i] &= recovery[i] >= latest - 0.5
        }
        let clear: [f32; 18] =
            std::array::from_fn(|i| nodes[i].action_minimum.min(timing.clear[i]));
        let best = (0..18)
            .filter(|i| allowed[*i])
            .map(|i| clear[i])
            .fold(f32::NEG_INFINITY, f32::max);
        if best.is_finite() {
            for i in 0..18 {
                allowed[i] &= clear[i] >= best - 1.
            }
        }
        if (0..18).any(|i| allowed[i] && !root_blocked[i]) {
            for i in 0..18 {
                allowed[i] &= !root_blocked[i]
            }
        }
        let short: [bool; 18] =
            std::array::from_fn(|i| nominal.free[i] && immediate[i] && !root_blocked[i]);
        if short.iter().any(|v| *v) {
            allowed = short;
            short_priority = true
        }
    }
    let mut scores: [f64; 18] = std::array::from_fn(|i| nodes[i].score);
    if !allowed.iter().any(|v| *v) {
        allowed[argmax(&scores, &[true; 18])] = true
    }
    let confidence = (confirmed.circles.len() + 4 * confirmed.bodies.len()) as f64
        / (scene.circles.len() + 4 * scene.bodies.len()).max(1) as f64;
    let reach = (fast * 8.).max(1.);
    let mut area = [0_f32; 18];
    let mut exits = [0_f64; 18];
    let mut measured = [false; 18];
    let mut moves = [[false; 8]; 18];
    for i in 0..18 {
        let p = nodes[i].p;
        let extent = [
            high[0].min(p[0] + reach) - low[0].max(p[0] - reach),
            high[1].min(p[1] + reach) - low[1].max(p[1] - reach),
        ];
        area[i] = (extent[0].max(0.) * extent[1].max(0.) / (4. * reach * reach)).clamp(0., 1.);
        for j in 0..8 {
            moves[i][j] = norm(sub(clip(point(p, velocities[10 + j], 6.), low, high), p)) > 0.1
        }
        exits[i] = moves[i].iter().filter(|v| **v).count() as f64 / 8.;
        scores[i] += (0.75 * (area[i] - 0.5)) as f64;
    }
    let mut probes = 0;
    if confirmed.circles.is_empty() && confirmed.bodies.is_empty() {
        measured = [true; 18]
    } else {
        for _ in 0..5 {
            let best = argmax(&scores, &allowed);
            if measured[best] {
                break;
            }
            let segments = confirmed.run(
                &[nodes[best].p; 8],
                &velocities[10..18],
                &[6.; 8],
                &[horizon; 8],
            );
            let value = segments
                .iter()
                .enumerate()
                .filter(|(j, s)| s.hit.is_infinite() && moves[best][*j])
                .count() as f64
                / 8.;
            scores[best] += 0.5 * confidence * (value - exits[best]);
            exits[best] = value;
            measured[best] = true;
            probes += 1;
        }
    }
    let best = argmax(&scores, &allowed);
    let late_risk = known_prefix.hit.is_finite();
    let best_contact = nominal
        .hit
        .iter()
        .map(|v| v.min(horizon))
        .fold(f32::NEG_INFINITY, f32::max);
    let rescue = fallback
        && !nominal.free.iter().any(|v| *v)
        && !late_risk
        && best_contact > late + 2.
        && nominal
            .clear
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            < -2.;
    let mut output = Vec::new();
    for i in 0..18 {
        let n = &nodes[i];
        output.extend_from_slice(&[
            1.,
            allowed[i] as u8 as f32,
            (n.minimum / 64.).clamp(-1., 1.),
            n.collision.min(horizon) / horizon,
            root_blocked[i] as u8 as f32,
            (n.goal / 1.5).clamp(0., 1.) as f32,
            (scores[i] / 8.).clamp(-1., 1.) as f32,
            ((scores[i] - scores[best]) * 8.).clamp(-16., 0.) as f32 / 16.,
        ]);
    }
    for n in &nodes {
        for p in &n.path {
            output.extend_from_slice(p)
        }
    }
    for n in &nodes {
        output.push(n.minimum)
    }
    for n in &nodes {
        output.push(n.collision)
    }
    for i in 0..18 {
        let n = &nodes[i];
        output.extend_from_slice(&[
            (safe[i] && !root_blocked[i]) as u8 as f32,
            fallback as u8 as f32,
            prefix.hit.is_finite() as u8 as f32,
            (n.action_minimum / 64.).clamp(-1., 1.),
            n.action_collision.min(horizon) / horizon,
            area[i],
            exits[i] as f32,
            measured[i] as u8 as f32,
            nominal.hit[i].min(horizon) / horizon,
        ]);
    }
    output.extend_from_slice(&[
        1.,
        fallback as u8 as f32,
        late_risk as u8 as f32,
        rescue as u8 as f32,
        1.,
        (best_contact / horizon).min(1.),
        (age / (1000. / 60. * horizon as f64)).min(1.) as f32,
        exits[best] as f32,
    ]);
    let count = |v: &[bool; 18]| v.iter().filter(|v| **v).count() as f64;
    let meta = vec![
        prefix.hit as f64,
        count(&timing.free),
        checked as f64,
        count(&safe),
        count(&allowed),
        fallback as u8 as f64,
        count(&nominal.free),
        rescue as u8 as f64,
        late_risk as u8 as f64,
        known_prefix.hit as f64,
        body_probe as u8 as f64,
        count(&immediate),
        probes as f64,
        confidence,
        area[best] as f64,
        exits[best],
        measured[best] as u8 as f64,
        short_priority as u8 as f64,
        best as f64,
    ];
    (output, meta)
}
