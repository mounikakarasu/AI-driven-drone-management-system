import React, { useState, useEffect } from 'react';
import { Activity, Battery, Navigation, ShieldAlert, Cpu } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Cell } from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';

const App = () => {
  const [data, setData] = useState(null);

  const [telemetry, setTelemetry] = useState({
    battery_level: 100,
    altitude: 50,
    velocity: 12,
    gps_signal: 9,
    obstacle_distance: 25
  });

  const fetchDecision = async () => {
    try {
      const API_URL = process.env.REACT_APP_API_URL || "";
const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(telemetry)
      });

      const result = await response.json();
      setData(result);

    } catch (err) {
      console.error("API Offline - Ensure Docker is running");
    }
  };

  useEffect(() => {
    fetchDecision();
  }, [telemetry]);

  const chartData = data?.logic_breakdown
    ? Object.entries(data.logic_breakdown).map(([name, value]) => ({ name, value }))
    : [];

  return (
    <div className="min-h-screen bg-slate-950 text-emerald-400 p-8 font-mono">

      {/* HUD Header */}
      <div className="flex justify-between items-center border-b border-emerald-900/50 pb-4 mb-8">
        <div className="flex items-center gap-3">
          <Activity className="animate-pulse" />
          <h1 className="text-2xl font-bold tracking-tighter text-emerald-500">
            DRONE_OS // COMMAND_CENTER
          </h1>
        </div>

        <div className="text-right text-xs">
          <p>SYSTEM_STATUS: <span className="text-emerald-500">OPTIMAL</span></p>
          <p>DOCKER_CONTAINER: <span className="text-blue-400">ACTIVE</span></p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Left: Telemetry Controls */}
        <div className="bg-slate-900/50 p-6 border border-emerald-900/30 rounded-lg">
          <h2 className="flex items-center gap-2 mb-6 text-sm opacity-70">
            <Navigation size={16}/> LIVE_CONTROLS
          </h2>

          {Object.keys(telemetry).map((key) => (
            <div key={key} className="mb-6">

              <div className="flex justify-between text-xs mb-2 capitalize">
                <span>{key.replace('_', ' ')}</span>
                <span>{telemetry[key]}</span>
              </div>

              <input
                type="range"
                className="w-full accent-emerald-500"
                min="0"
                max={key === 'altitude' ? 500 : 100}
                value={telemetry[key]}
                onChange={(e) =>
                  setTelemetry({
                    ...telemetry,
                    [key]: parseFloat(e.target.value)
                  })
                }
              />

            </div>
          ))}
        </div>

        {/* Center/Right: AI Decision HUD */}
        <div className="lg:col-span-2 space-y-6">

          <AnimatePresence mode="wait">
            <motion.div
              key={data?.decision}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className={`p-10 border-2 rounded-xl text-center ${
                data?.decision === 'RETURN_HOME'
                  ? 'border-red-500 bg-red-950/20'
                  : 'border-emerald-500 bg-emerald-950/10'
              }`}
            >

              <p className="text-xs uppercase tracking-[0.3em] mb-2 opacity-60">
                AI Neural Inference
              </p>

              <h2 className="text-6xl font-black mb-4">
                {data?.decision || "INITIALIZING..."}
              </h2>

              <div className="flex justify-center gap-8 text-sm">
                <div className="flex items-center gap-2">
                  <Cpu size={16}/>
                  CONFIDENCE: {data ? data.confidence.toFixed(0) : 0}%
                </div>

                <div className="flex items-center gap-2">
                  <ShieldAlert size={16}/>
                  PRIMARY_FACTOR: {data?.primary_factor || "N/A"}
                </div>
              </div>

            </motion.div>
          </AnimatePresence>

          {/* Explainable AI Graph */}
          <div className="bg-slate-900/50 p-6 border border-emerald-900/30 rounded-lg h-72">

            <h2 className="text-xs opacity-50 mb-4 uppercase tracking-widest">
              Decision Logic Breakdown (SHAP)
            </h2>

            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ left: 30, right: 30 }}>
                <XAxis type="number" hide domain={['auto', 'auto']} />
                <YAxis dataKey="name" type="category" stroke="#059669" fontSize={10} width={120} />

                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell
                      key={index}
                      fill={entry.value > 0 ? '#10b981' : '#ef4444'}
                    />
                  ))}
                </Bar>

              </BarChart>
            </ResponsiveContainer>

          </div>

          {/* Black Box Recorder Status */}
          <div className="mt-6 p-4 bg-emerald-950/10 border border-emerald-900/30 rounded-lg text-[10px] opacity-50">
            <p>{'>'} BLACK_BOX_RECORDING_ACTIVE: Writing to /app/flight_log.csv...</p>
            <p>{'>'} LATENCY: 14ms | SHAP_ENGINE: TREE_EXPLORER | UPTIME: {Math.floor(performance.now()/1000)}s</p>
          </div>

        </div>
      </div>
    </div>
  );
};

export default App;
