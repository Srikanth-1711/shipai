import React, { useEffect, useState } from 'react';
import type { CanvasNode as CanvasNodeType } from '../../types';
import { 
  User, Brain, Database, Search, Filter, 
  Zap, ShieldAlert, Sparkles, Settings, Activity 
} from 'lucide-react';

const ICON_MAP: Record<string, React.ReactNode> = {
  'userNode': <User size={24} />,
  'orchestratorNode': <Brain size={24} />,
  'vector_dbNode': <Database size={24} />,
  'search_typeNode': <Search size={24} />,
  'rerankerNode': <Filter size={24} />,
  'cacheNode': <Zap size={24} />,
  'piiNode': <ShieldAlert size={24} />,
  'llmNode': <Sparkles size={24} />,
  'workerNode': <Settings size={24} />,
  'monitorNode': <Activity size={24} />,
  'infraNode': <Settings size={24} />
};

const COLOR_MAP: Record<string, string> = {
  'userNode': 'text-gray-400 border-gray-600',
  'orchestratorNode': 'text-purple-400 border-purple-500 bg-purple-900/20',
  'vector_dbNode': 'text-blue-400 border-blue-500 bg-blue-900/20',
  'search_typeNode': 'text-blue-400 border-blue-500 bg-blue-900/20',
  'rerankerNode': 'text-amber-400 border-amber-500 bg-amber-900/20',
  'cacheNode': 'text-amber-400 border-amber-500 bg-amber-900/20',
  'piiNode': 'text-red-400 border-red-500 bg-red-900/20',
  'llmNode': 'text-purple-400 border-purple-500 bg-purple-900/20',
  'workerNode': 'text-green-400 border-green-500 bg-green-900/20',
  'monitorNode': 'text-green-400 border-green-500 bg-green-900/20',
  'infraNode': 'text-gray-400 border-gray-500 bg-gray-900/20'
};

interface CanvasNodeProps {
  node: CanvasNodeType;
  pulseComplete: boolean;
}

export const CanvasNode: React.FC<CanvasNodeProps> = ({ node, pulseComplete }) => {
  const [appeared, setAppeared] = useState(false);
  const [pulsing, setPulsing] = useState(false);

  useEffect(() => {
    // Trigger entrance animation on mount
    const timer = setTimeout(() => setAppeared(true), 50);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (pulseComplete) {
      setPulsing(true);
      const timer = setTimeout(() => setPulsing(false), 600);
      return () => clearTimeout(timer);
    }
  }, [pulseComplete]);

  const confirmed = node.status === 'confirmed';
  
  // Base colors
  let colors = COLOR_MAP[node.type] || 'text-gray-400 border-gray-500 bg-gray-800';
  if (!confirmed) {
    colors = 'text-gray-500 border-gray-600 border-dashed bg-gray-900';
  }

  const icon = ICON_MAP[node.type] || <Settings size={24} />;

  // Pulse effect class
  const pulseClass = pulsing ? 'ring-4 ring-white/50 scale-110' : '';

  return (
    <g 
      transform={`translate(${node.x || 0}, ${node.y || 0})`}
      style={{
        opacity: confirmed ? 1 : 0.4,
        transition: 'opacity 0.4s ease, transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
        transform: `translate(${node.x || 0}px, ${node.y || 0}px) scale(${appeared ? 1 : 0.5})`
      }}
    >
      <foreignObject x="-75" y="-35" width="150" height="70">
        <div 
          className={`w-full h-full flex flex-col items-center justify-center rounded-xl border-2 shadow-lg transition-all duration-300 ${colors} ${pulseClass}`}
        >
          <div className="mb-1">{icon}</div>
          <div className="text-[10px] font-semibold text-center leading-tight px-2 whitespace-nowrap overflow-hidden text-ellipsis max-w-[140px]">
            {node.label}
          </div>
        </div>
      </foreignObject>
    </g>
  );
};
