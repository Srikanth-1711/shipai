import { useMemo } from 'react';
import type { CanvasNode } from '../../types';

export const useCanvasLayout = (nodes: CanvasNode[], width: number, height: number) => {
  return useMemo(() => {
    // Fixed layout grid based on RAG architectural typical flow
    const layoutMap: Record<string, { x: number, y: number }> = {
      'userNode': { x: width / 2, y: 50 },
      'orchestratorNode': { x: width / 2, y: 150 },
      
      // Data Layer
      'vector_dbNode': { x: width / 2 + 150, y: 150 },
      'search_typeNode': { x: width / 2 + 150, y: 250 },
      
      // Inference Layer
      'rerankerNode': { x: width / 2, y: 250 },
      'llmNode': { x: width / 2, y: 350 },
      'cacheNode': { x: width / 2 - 150, y: 350 },
      
      // Security
      'piiNode': { x: width / 2 - 150, y: 150 },
      
      // Infra Layer
      'infraNode': { x: width / 2, y: 450 },
      'monitorNode': { x: width / 2 + 150, y: 450 },
      'workerNode': { x: width / 2 - 150, y: 450 }
    };

    return nodes.map(node => {
      const pos = layoutMap[node.type] || { x: width / 2, y: height / 2 };
      return { ...node, x: pos.x, y: pos.y };
    });
  }, [nodes, width, height]);
};
