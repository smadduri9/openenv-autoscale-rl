import { useEffect, useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
  ResponsiveContainer
} from "recharts";

const SPEEDS = [
  { label: "0.5x", ms: 1000 },
  { label: "1x", ms: 600 },
  { label: "2x", ms: 300 },
  { label: "4x", ms: 120 }
];

const FAMILY_LABELS = {
  steady: "Steady Load",
  sustained_spike: "Sustained Spike",
  transient_spike: "Transient Spike",
  cyclical: "Cyclical Demand",
  drifting_noisy: "Drifting Demand",
  traffic_spike: "Traffic Spike",
  bad_deploy: "Bad Deploy",
  dependency_slowdown: "Dependency Slowdown"
};

function toTitle(value) {
  return String(value || "")
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function getScenarioLabel(family) {
  return FAMILY_LABELS[family] || toTitle(family);
}

function getScenarioNumber(traceId) {
  const match = String(traceId || "").match(/_(\d+)$/);
  if (!match) return null;
  return Number(match[1]) + 1;
}

function getTraceDisplayName(trace) {
  const scenario = getScenarioLabel(trace?.family);
  const number = getScenarioNumber(trace?.trace_id);
  if (number !== null && Number.isFinite(number)) return `${scenario} ${number}`;
  return scenario;
}

function Panel({ title, frame, totalReward }) {
  if (!frame) return <div className="card">No frame data</div>;
  const badges = [];
  if ("rate_limit_enabled" in frame && frame.rate_limit_enabled) badges.push("Rate Limit Enabled");
  if ("bad_deploy_active" in frame && frame.bad_deploy_active) badges.push("Bad Deploy Active");
  if ("dependency_slowdown_active" in frame && frame.dependency_slowdown_active) {
    badges.push("Dependency Slowdown");
  }
  if ("rollback_pending_steps" in frame && Number(frame.rollback_pending_steps) > 0) {
    badges.push(`Rollback Pending: ${frame.rollback_pending_steps}`);
  }
  return (
    <div className="card">
      <div className="card-title-row">
        <h3>{title}</h3>
        <div className="badges">
          {badges.map((b) => (
            <span key={b} className="badge">{b}</span>
          ))}
        </div>
      </div>
      <div className="stat-grid">
        <div><span>Timestep</span>{frame.timestep}</div>
        <div><span>Action</span>{frame.action}</div>
        <div><span>Step Reward</span>{frame.step_reward.toFixed(4)}</div>
        <div><span>Cumulative Reward</span>{frame.cumulative_reward.toFixed(4)}</div>
        <div><span>Incoming RPS</span>{frame.incoming_rps.toFixed(2)}</div>
        <div><span>Ready Pods</span>{frame.ready_pods}</div>
        <div><span>Queue Depth</span>{frame.queue_depth.toFixed(2)}</div>
        <div><span>P95 Latency (ms)</span>{frame.p95_latency_ms.toFixed(2)}</div>
        <div><span>Previous Action</span>{frame.previous_action}</div>
        <div><span>Error Rate</span>{frame.error_rate.toFixed(4)}</div>
        <div><span>Episode Total Reward</span>{totalReward.toFixed(4)}</div>
      </div>
    </div>
  );
}

function SummaryCards({ replay, mode }) {
  if (!replay) return null;
  const h = replay.heuristic;
  const r = replay.rl;
  const hMetrics = h?.metrics || {};
  const rMetrics = r?.metrics || {};
  const scenarioName = getTraceDisplayName(replay);
  return (
    <div className="summary-grid">
      <div className="summary-card">
        <p className="summary-label">Scenario</p>
        <h2>{scenarioName}</h2>
        <p className="summary-subtext">Trace ref: {replay.trace_id}</p>
      </div>
      <div className="summary-card">
        <p className="summary-label">Heuristic Total Reward</p>
        <h2>{(h?.total_reward || 0).toFixed(4)}</h2>
        <p>Mean latency: {Number(hMetrics.mean_latency_ms || 0).toFixed(2)} ms</p>
      </div>
      <div className="summary-card">
        <p className="summary-label">RL Total Reward</p>
        <h2>{mode === "heuristic" || !r ? "N/A" : r.total_reward.toFixed(4)}</h2>
        <p>
          Mean latency:{" "}
          {mode === "heuristic" || !r ? "N/A" : `${Number(rMetrics.mean_latency_ms || 0).toFixed(2)} ms`}
        </p>
      </div>
      <div className="summary-card">
        <p className="summary-label">Policy Delta</p>
        <h2>
          {mode === "compare" && r
            ? `${(r.total_reward - h.total_reward).toFixed(4)} reward delta`
            : "Switch to Heuristic vs RL"}
        </h2>
        <p>
          {mode === "compare" && r
            ? `${(Number(rMetrics.drop_fraction || 0) * 100).toFixed(2)}% vs ${(Number(hMetrics.drop_fraction || 0) * 100).toFixed(2)}% drop rate`
            : "Comparison metrics appear here once both policies are visible."}
        </p>
      </div>
    </div>
  );
}

function ChartCard({ title, data, lines, height = 220 }) {
  return (
    <div className="chart-card">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="step" />
          <YAxis />
          <Tooltip />
          <Legend />
          {lines.map((line) => (
            <Line
              key={line.dataKey}
              dataKey={line.dataKey}
              name={line.name}
              stroke={line.stroke}
              strokeDasharray={line.strokeDasharray}
              strokeWidth={line.strokeWidth || 2.8}
              activeDot={{ r: 5 }}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function App() {
  const [traces, setTraces] = useState([]);
  const [mode, setMode] = useState("heuristic");
  const [traceIndex, setTraceIndex] = useState(0);
  const [replay, setReplay] = useState(null);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(SPEEDS[1].ms);
  const [rlAvailable, setRlAvailable] = useState(false);

  useEffect(() => {
    fetch("/api/traces")
      .then((r) => r.json())
      .then((d) => {
        setTraces(d.traces || []);
        setRlAvailable(Boolean(d.rl_available));
        if (!d.rl_available) setMode("heuristic");
      });
  }, []);

  useEffect(() => {
    if (!traces.length) return;
    fetch(`/api/replay?trace_index=${traceIndex}&mode=${mode}`)
      .then((r) => r.json())
      .then((d) => {
        setReplay(d);
        setStepIdx(0);
        setPlaying(false);
      })
      .catch(() => {
        setReplay(null);
      });
  }, [traces, traceIndex, mode]);

  useEffect(() => {
    if (!playing || !replay) return undefined;
    const frames = replay.heuristic?.frames || [];
    const timer = setInterval(() => {
      setStepIdx((prev) => {
        if (prev >= frames.length - 1) {
          setPlaying(false);
          return frames.length - 1;
        }
        return prev + 1;
      });
    }, speedMs);
    return () => clearInterval(timer);
  }, [playing, speedMs, replay]);

  const hFrame = replay?.heuristic?.frames?.[stepIdx];
  const rFrame = replay?.rl?.frames?.[stepIdx];

  const chartData = useMemo(() => {
    if (!replay?.heuristic?.frames) return [];
    return replay.heuristic.frames.map((h, i) => {
      const r = replay.rl?.frames?.[i];
      return {
        step: h.timestep,
        incoming_rps: h.incoming_rps,
        ready_pods_h: h.ready_pods,
        queue_depth_h: h.queue_depth,
        latency_h: h.p95_latency_ms,
        cumulative_reward_h: h.cumulative_reward,
        ready_pods_r: r ? r.ready_pods : null,
        queue_depth_r: r ? r.queue_depth : null,
        latency_r: r ? r.p95_latency_ms : null,
        cumulative_reward_r: r ? r.cumulative_reward : null
      };
    });
  }, [replay]);

  const maxStep = Math.max(0, (replay?.heuristic?.frames?.length || 1) - 1);
  const traceCount = traces.length;
  const selectedTraceMeta = traces.find((t) => t.index === traceIndex);
  const selectedTraceName = selectedTraceMeta ? getTraceDisplayName(selectedTraceMeta) : "-";
  const selectedScenario = selectedTraceMeta ? getScenarioLabel(selectedTraceMeta.family) : "-";

  return (
    <div className="app">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">OpenEnv Demo Experience</p>
          <h1>OpenEnv Autoscaling Control Center</h1>
          <p className="subtitle">
            Interactive simulation of policy behavior under dynamic service load.
          </p>
          <p className="subsubtitle">
            Analyze decision quality, latency-pressure response, and reward trajectory with synchronized policy replay.
          </p>
        </div>
        <div className="hero-pill">Live Replay</div>
      </header>
      <div className="controls">
        <div className="trace-nav">
          <button onClick={() => setTraceIndex((v) => Math.max(0, v - 1))} disabled={traceIndex <= 0}>
            Prev
          </button>
          <button
            onClick={() => setTraceIndex((v) => Math.min(traceCount - 1, v + 1))}
            disabled={traceIndex >= traceCount - 1}
          >
            Next
          </button>
        </div>
        <label>
          Scenario:
          <select className="pretty-select" value={traceIndex} onChange={(e) => setTraceIndex(Number(e.target.value))}>
            {traces.map((t) => (
              <option key={t.index} value={t.index}>
                {getTraceDisplayName(t)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Policy View:
          <select className="pretty-select" value={mode} onChange={(e) => setMode(e.target.value)} disabled={!rlAvailable}>
            <option value="heuristic">Heuristic Baseline</option>
            <option value="rl" disabled={!rlAvailable}>RL Policy</option>
            <option value="compare" disabled={!rlAvailable}>Heuristic vs RL (Showcase)</option>
          </select>
        </label>
        <button onClick={() => setPlaying(true)}>Play</button>
        <button onClick={() => setPlaying(false)}>Pause</button>
        <button onClick={() => { setPlaying(false); setStepIdx(0); }}>Reset</button>
        <label>
          Playback Speed:
          <select className="pretty-select" value={speedMs} onChange={(e) => setSpeedMs(Number(e.target.value))}>
            {SPEEDS.map((s) => (
              <option key={s.label} value={s.ms}>{s.label}</option>
            ))}
          </select>
        </label>
        <label>
          Step:
          <input
            type="range"
            min={0}
            max={maxStep}
            value={stepIdx}
            onChange={(e) => setStepIdx(Number(e.target.value))}
          />
          {stepIdx}
        </label>
      </div>
      <div className="trace-count-note">
        Replay catalog: <strong>{traceCount}</strong> trace{traceCount === 1 ? "" : "s"}
        {traceCount <= 1 && (
          <span>
            {" "}
            — only one trace is bundled in the loaded replay file. Include additional traces for a broader showcase.
          </span>
        )}
      </div>

      <SummaryCards replay={replay} mode={mode} />

      <div className="context-row">
        <div className="context-card">
          <strong>Scenario:</strong> {selectedScenario} &nbsp; | &nbsp;
          <strong>Run:</strong> {selectedTraceName} &nbsp; | &nbsp;
          <strong>Trace ID:</strong> {replay?.trace_id || "-"} &nbsp; | &nbsp;
          <strong>RL Track Available:</strong> {String(replay?.rl_available ?? false)}
        </div>
      </div>

      <div className={mode === "compare" ? "grid2" : "grid1"}>
        <Panel title="Heuristic" frame={hFrame} totalReward={replay?.heuristic?.total_reward || 0} />
        {mode === "compare" && (
          <Panel title="RL" frame={rFrame} totalReward={replay?.rl?.total_reward || 0} />
        )}
      </div>

      <ChartCard
        title="Incoming RPS Over Time"
        data={chartData}
        lines={[{ dataKey: "incoming_rps", name: "incoming_rps", stroke: "#38bdf8", strokeWidth: 2.6 }]}
      />
      <ChartCard
        title="Ready Pods Over Time"
        data={chartData}
        lines={[
          { dataKey: "ready_pods_h", name: "heuristic", stroke: "#00e5ff", strokeWidth: 3.2 },
          ...(mode === "compare"
            ? [{ dataKey: "ready_pods_r", name: "rl", stroke: "#ff2d95", strokeDasharray: "8 4", strokeWidth: 3.2 }]
            : []),
        ]}
      />
      <ChartCard
        title="Queue Depth Over Time"
        data={chartData}
        lines={[
          { dataKey: "queue_depth_h", name: "heuristic", stroke: "#22d3ee", strokeWidth: 3.2 },
          ...(mode === "compare"
            ? [{ dataKey: "queue_depth_r", name: "rl", stroke: "#f43f5e", strokeDasharray: "8 4", strokeWidth: 3.2 }]
            : []),
        ]}
      />
      <ChartCard
        title="P95 Latency Over Time"
        data={chartData}
        lines={[
          { dataKey: "latency_h", name: "heuristic", stroke: "#67e8f9", strokeWidth: 3.2 },
          ...(mode === "compare"
            ? [{ dataKey: "latency_r", name: "rl", stroke: "#fb7185", strokeDasharray: "8 4", strokeWidth: 3.2 }]
            : []),
        ]}
      />
      <ChartCard
        title="Cumulative Reward Over Time"
        data={chartData}
        lines={[
          { dataKey: "cumulative_reward_h", name: "heuristic", stroke: "#22d3ee", strokeWidth: 3.2 },
          ...(mode === "compare"
            ? [{ dataKey: "cumulative_reward_r", name: "rl", stroke: "#ff4d6d", strokeDasharray: "8 4", strokeWidth: 3.2 }]
            : []),
        ]}
      />
    </div>
  );
}
