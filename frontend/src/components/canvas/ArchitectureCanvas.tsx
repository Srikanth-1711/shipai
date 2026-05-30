import React, { useRef, useState, useEffect } from 'react';
import { useShipAIStore } from '../../store/useShipAIStore';
import { CanvasNode } from './CanvasNode';
import { CanvasEdge } from './CanvasEdge';
import { useCanvasLayout } from './useCanvasLayout';
import { Download } from 'lucide-react';

export const ArchitectureCanvas: React.FC = () => {
  const { canvasNodes, canvasEdges, architectureComplete } = useShipAIStore();
  const containerRef = useRef<HTMLDivElement>(null);
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    if (containerRef.current) {
      setDimensions({
        width: containerRef.current.clientWidth,
        height: containerRef.current.clientHeight
      });
    }
  }, []);

  const positionedNodes = useCanvasLayout(canvasNodes, dimensions.width, dimensions.height);

  return (
    <div className="w-full h-full bg-gray-900 relative" ref={containerRef}>
      {/* Background Grid */}
      <div 
        className="absolute inset-0 pointer-events-none opacity-20"
        style={{
          backgroundImage: 'linear-gradient(#374151 1px, transparent 1px), linear-gradient(90deg, #374151 1px, transparent 1px)',
          backgroundSize: '20px 20px'
        }}
      />
      
      {/* SVG Canvas */}
      {dimensions.width > 0 && (
        <svg className="w-full h-full absolute inset-0 z-10 pointer-events-none">
          {/* Edges rendered first so they sit behind nodes */}
          {canvasEdges.map(edge => (
            <CanvasEdge key={edge.id} edge={edge} nodes={positionedNodes} />
          ))}
          
          {/* Nodes rendered on top */}
          {positionedNodes.map(node => (
            <CanvasNode 
              key={node.id} 
              node={node} 
              pulseComplete={architectureComplete} 
            />
          ))}
        </svg>
      )}

      {/* Completion Badge */}
      {architectureComplete && (
        <div className="absolute top-4 right-4 z-20 flex flex-col gap-2">
          <div className="bg-green-900/80 border border-green-500 text-green-300 px-4 py-2 rounded-lg shadow-lg font-medium text-sm flex items-center gap-2">
            Architecture Complete
          </div>
          <button className="bg-gray-800 hover:bg-gray-700 border border-gray-600 text-gray-200 px-4 py-2 rounded-lg shadow-lg font-medium text-sm flex items-center gap-2 transition-colors">
            <Download size={16} />
            Export Diagram
          </button>
        </div>
      )}
      
      {canvasNodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-sm z-0">
          Awaiting requirements to begin architecture design...
        </div>
      )}
    </div>
  );
};
