import React, { useState } from 'react';
import { useShipAIStore } from '../../store/useShipAIStore';

const optionsMap: Record<string, string[]> = {
  vector_db: ['chroma', 'pgvector', 'milvus', 'qdrant'],
  search_type: ['dense', 'hybrid', 'sparse'],
  cache: ['none', 'semantic', 'redis_basic'],
  pii: ['none', 'presidio', 'custom'],
  infra_tier: ['minimal', 'standard', 'enterprise'],
  framework: ['langgraph', 'llamaindex', 'haystack']
};

export const ConfigEditor: React.FC = () => {
  const { configJson, sessionId, setProjectData, updateCanvasNode } = useShipAIStore();
  const [modifiedFields, setModifiedFields] = useState<Set<string>>(new Set());

  if (!configJson || Object.keys(configJson).length === 0) return null;

  const handleChange = async (field: string, newValue: string) => {
    // 1. Update local configJson state
    const newConfig = { ...configJson, [field]: newValue };
    setProjectData(useShipAIStore.getState().projectPath!, newConfig, sessionId!);
    
    // 2. Mark as modified
    setModifiedFields(prev => new Set(prev).add(field));
    
    // 3. Update Canvas
    updateCanvasNode(`node_${field}`, { 
      label: newValue, 
      status: 'confirmed' 
    });

    // 4. Send Feedback API
    if (sessionId) {
      try {
        await fetch(`http://${window.location.hostname}:8000/api/feedback/modify/${sessionId}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ [field]: newValue })
        });
      } catch (err) {
        console.error("Failed to send feedback", err);
      }
    }
  };

  return (
    <div className="p-4 border-t border-gray-800 bg-gray-900/50">
      <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Architecture Decisions</h3>
      <div className="space-y-3">
        {Object.entries(configJson).map(([field, value]) => {
          if (!optionsMap[field]) return null;
          
          let valStr = typeof value === 'object' ? value.choice : value;
          const isModified = modifiedFields.has(field);

          return (
            <div key={field} className="flex items-center justify-between">
              <label className="text-sm text-gray-300 capitalize">{field.replace('_', ' ')}</label>
              <div className="flex items-center gap-2">
                {isModified && <span className="text-[10px] text-amber-500 font-medium">You changed this</span>}
                <select 
                  value={valStr}
                  onChange={(e) => handleChange(field, e.target.value)}
                  className={`text-sm bg-gray-800 border rounded px-2 py-1 outline-none transition-colors ${
                    isModified ? 'border-amber-500 text-amber-400' : 'border-gray-700 text-gray-200'
                  }`}
                >
                  {optionsMap[field].map(opt => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
