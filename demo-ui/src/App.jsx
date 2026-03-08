import { useEffect, useMemo, useState } from "react";
import {
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
  Legend
} from "recharts";

const SPEED_OPTIONS = [
  { label: "0.5x", ms: 1200 },
  { label: "1x", ms: 700 },
  { label: "2x", ms: 350 },
  { label: "4x", ms: 150 }
];

function PolicyPanel({ title, frame, totalReward }) {
  if (!frame) return null;
  return (
    <div className="panel">
      <h3>{title}</h3>
      <div className="metric-grid">
        <div><span>Action:</span>{frame.action}</div>
        <div><span>Step:</span>{frame.timestep}</div>
        <div><span>Step Reward:</span>{frame.step_reward.toFixed(4)}</div>
        <div><span>Cumulative Reward:</span>{frame.cumulative_reward.toFixed(4)}</div>
        <div><span>Incoming RPS:</span>{frame.incoming_rps.toFixed(2)}</div>
        <div><span>Ready Pods:</span>{frame.ready_pods}</div>
        <div><span>Queue Depth:</span>{frame.queue_depth.toFixed(2)}</div>
        <div><span>P95 Latency:</span>{frame.p95_latency_ms.toFixed(2)}</div>
        <div><span>Error Rate:</span>{frame.error_rate.toFixed(4)}</div>
        <div><span>Drop Estimate:</span>{frame.drop_estimate.toFixed(3)}</div>
        <div><span>Rate Limit:</span>{String(frame.rate_limit_enabled)}</div>
        <div><span>Bad Deploy:</span>{String(frame.bad_deploy_active)}</div>
        <div><span>Dependency Slowdown:</span>{String(frame.dependency_slowdown_active)}</div>
        <div><span>Rollback Pending:</span>{frame.rollback_pending_steps}</div>
        <div><span>Total Reward (episode):</span>{totalReward.toFixed(4)}</div>
      </div>
    </div>
  );
}

export default function App() {
  const [payload, setPayload] = useState(null);
  const [traceIndex, setTraceIndex] = useState(0);
  const [stepIndex, setStepIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speedMs, setSpeedMs] = useState(SPEED_OPTIONS[1].ms);

  useEffect(() => {
    fetch("/api/replays")
      .then((r) => r.json())
      .then((d) => setPayload(d))
      .catch((err) => console.error("Failed to load replay payload", err));
  }, []);

  const selectedTrace = payload?.traces?.[traceIndex];
  const maxSteps = selectedTrace ? Math.min(selectedTrace.heuristic.frames.length, selectedTrace.rl.frames.length) : 0;

  useEffect(() => {
    if (!playing || maxSteps <= 0) return undefined;
    const timer = setInterval(() => {
      setStepIndex((prev) => {
        if (prev >= maxSteps - 1) {
          setPlaying(false);
          return maxSteps - 1;
        }
        return prev + 1;
      });
    }, speedMs);
    return () => clearInterval(timer);
  }, [playing, speedMs, maxSteps]);

  useEffect(() => {
    setStepIndex(0);
    setPlaying(false);
  }, [traceIndex]);

  const chartData = useMemo(() => {
    if (!selectedTrace) return [];
    return selectedTrace.heuristic.frames.map((hFrame, idx) => {
      const rFrame = selectedTrace.rl.frames[idx];
      return {
        step: hFrame.timestep,
        heuristic_queue: hFrame.queue_depth,
        rl_queue: rFrame.queue_depth,
        heuristic_latency: hFrame.p95_latency_ms,
        rl_latency: rFrame.p95_latency_ms,
        heuristic_reward: hFrame.cumulative_reward,
        rl_reward: rFrame.cumulative_reward
      };
    });
  }, [selectedTrace]);

  if (!payload) {
    return <div className="container">Loading replay payload...</div>;
  }

  const hFrame = selectedTrace?.heuristic.frames?.[stepIndex];
  const rFrame = selectedTrace?.rl.frames?.[stepIndex];

  return (
    <div className="container">
      <h1>Incident Horizon 20: Heuristic vs RL Replay</h1>
      <div className="controls">
        <label>
          Trace / Scenario:
          <select value={traceIndex} onChange={(e) => setTraceIndex(Number(e.target.value))}>
            {payload.traces.map((t, idx) => (
              <option key={t.trace_id} value={idx}>
                {t.trace_id} ({t.family})
              </option>
            ))}
          </select>
        </label>
        <button onClick={() => setPlaying(true)} disabled={playing}>Play</button>
        <button onClick={() => setPlaying(false)}>Pause</button>
        <button onClick={() => { setPlaying(false); setStepIndex(0); }}>Reset</button>
        <label>
          Speed:
          <select value={speedMs} onChange={(e) => setSpeedMs(Number(e.target.value))}>
            {SPEED_OPTIONS.map((s) => (
              <option key={s.label} value={s.ms}>{s.label}</option>
            ))}
          </select>
        </label>
        <label>
          Timestep:
          <input
            type="range"
            min={0}
            max={Math.max(0, maxSteps - 1)}
            value={stepIndex}
            onChange={(e) => setStepIndex(Number(e.target.value))}
          />
          {hFrame?.timestep ?? 0}
        </label>
      </div>

      <div className="split">
        <PolicyPanel title="Heuristic" frame={hFrame} totalReward={selectedTrace.heuristic.total_reward} />
        <PolicyPanel title="RL" frame={rFrame} totalReward={selectedTrace.rl.total_reward} />
      </div>

      <div className="chart-row">
        <div className="chart-card">
          <h3>Queue Depth</h3>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="heuristic_queue" stroke="#1f77b4" dot={false} />
              <Line type="monotone" dataKey="rl_queue" stroke="#ff7f0e" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div className="chart-card">
          <h3>P95 Latency</h3>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="heuristic_latency" stroke="#2ca02c" dot={false} />
              <Line type="monotone" dataKey="rl_latency" stroke="#d62728" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="chart-row">
        <div className="chart-card">
          <h3>Cumulative Reward</h3>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="step" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="heuristic_reward" stroke="#9467bd" dot={false} />
              <Line type="monotone" dataKey="rl_reward" stroke="#8c564b" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
