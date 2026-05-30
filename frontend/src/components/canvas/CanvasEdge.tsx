import React, { useEffect, useState } from 'react';
import type { CanvasEdge as CanvasEdgeType, CanvasNode } from '../../types';

interface CanvasEdgeProps {
  edge: CanvasEdgeType;
  nodes: CanvasNode[];
}

export const CanvasEdge: React.FC<CanvasEdgeProps> = ({ edge, nodes }) => {
  const [drawn, setDrawn] = useState(false);

  useEffect(() => {
    // Small delay to allow nodes to appear first
    const timer = setTimeout(() => setDrawn(true), 300);
    return () => clearTimeout(timer);
  }, []);

  const sourceNode = nodes.find(n => n.id === edge.source);
  const targetNode = nodes.find(n => n.id === edge.target);

  if (!sourceNode || !targetNode) return null;
  if (sourceNode.x === undefined || sourceNode.y === undefined) return null;
  if (targetNode.x === undefined || targetNode.y === undefined) return null;

  // We draw from the center/bottom of source to center/top of target
  // A simple cubic bezier curve works nicely for flow layout
  const sx = sourceNode.x;
  const sy = sourceNode.y + 35; // offset by half height of foreignObject
  const tx = targetNode.x;
  const ty = targetNode.y - 35;

  const path = `M ${sx} ${sy} C ${sx} ${(sy + ty) / 2}, ${tx} ${(sy + ty) / 2}, ${tx} ${ty}`;

  let strokeDasharray = "1000";
  if (edge.type === 'async/queue') strokeDasharray = "5, 5";
  if (edge.type === 'monitoring') strokeDasharray = "2, 4";

  return (
    <g>
      {/* Invisible thicker path for easier hover (if needed later) */}
      <path d={path} fill="none" stroke="transparent" strokeWidth="10" />
      
      {/* The actual visible edge */}
      <path 
        d={path} 
        fill="none" 
        stroke="rgba(107, 114, 128, 0.6)" 
        strokeWidth="2"
        strokeDasharray={edge.type === 'data flow' ? strokeDasharray : "1000"} 
        strokeDashoffset={drawn ? 0 : 1000}
        style={{
          transition: 'stroke-dashoffset 0.8s ease-in-out',
        }}
      />
      
      {/* If it's dashed, we draw a solid line animating first, then apply dashes 
          (Simplified here: just CSS transition on a solid line, or we fake the dash animation) */}
    </g>
  );
};
