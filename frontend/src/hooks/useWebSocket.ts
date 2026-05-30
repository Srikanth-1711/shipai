import { useEffect, useRef } from 'react';
import { useShipAIStore } from '../store/useShipAIStore';
import type { WSEvent, AgentType } from '../types';

export function useWebSocket(url: string) {
  const wsRef = useRef<WebSocket | null>(null);
  
  const {
    addMessage,
    appendToken,
    setStreaming,
    setCurrentAgent,
    updateRequirement,
    addCanvasNode,
    addCanvasEdge,
    setProjectData,
    setArchitectureComplete,
  } = useShipAIStore();

  useEffect(() => {
    let ws: WebSocket;
    let reconnectTimer: any;

    function connect() {
      ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        const data: WSEvent = JSON.parse(event.data);
        
        switch (data.type) {
          case 'agent_change':
            // The payload comes from AGENT_CONFIG: { color: string, label: string }
            // We map this roughly to AgentType if needed, or just let UI show the label
            const label = data.payload.label.toLowerCase();
            let newAgent: AgentType = 'discovery';
            if (label.includes('architect')) newAgent = 'architect';
            if (label.includes('builder')) newAgent = 'builder';
            if (label.includes('explainer')) newAgent = 'explainer';
            
            setCurrentAgent(newAgent);
            
            // Create a new message bubble for this agent
            addMessage({
              id: Date.now().toString() + Math.random(),
              role: newAgent,
              content: '',
              timestamp: new Date(),
              isStreaming: true
            });
            setStreaming(true);
            break;
            
          case 'token':
            appendToken(data.payload.text);
            break;
            
          case 'requirement':
            updateRequirement(data.payload.field, data.payload.value);
            break;
            
          case 'decision':
            // Add a node for the decision
            const nodeId = `node_${data.payload.component}`;
            addCanvasNode({
              id: nodeId,
              type: `${data.payload.component}Node`,
              label: data.payload.choice,
              status: 'confirmed'
            });
            
            // Connect it to the Orchestrator (example logic)
            addCanvasEdge({
              id: `edge_orchestrator_${nodeId}`,
              source: 'node_orchestrator',
              target: nodeId,
              type: 'data flow'
            });
            break;
            
          case 'file_generated':
            // We will update the file tree
            // Simplified: just push to array (in a real app, parse paths into tree structure)
            // Store raw list in Zustand for now and let component parse it
            useShipAIStore.setState((state) => ({
               fileTree: [...state.fileTree, {
                   name: data.payload.name,
                   type: 'file',
                   path: data.payload.path,
                   highlight: data.payload.highlighted
               }]
            }));
            break;
            
          case 'complete':
            setStreaming(false);
            if (data.payload.project_path) {
              setArchitectureComplete(true);
              setProjectData(data.payload.project_path, data.payload.config_json, data.payload.session_id);
            }
            
            // Set streaming to false on the last message
            useShipAIStore.setState((state) => ({
               messages: state.messages.map((m, i) => 
                 i === state.messages.length - 1 ? { ...m, isStreaming: false } : m
               )
            }));
            break;
            
          case 'error':
            setStreaming(false);
            addMessage({
              id: Date.now().toString(),
              role: 'explainer',
              content: `Error: ${data.payload.message}`,
              timestamp: new Date(),
              isStreaming: false
            });
            break;
        }
      };

      ws.onclose = () => {
        console.log("WebSocket disconnected. Reconnecting in 3s...");
        reconnectTimer = setTimeout(connect, 3000);
      };

      ws.onerror = (err) => {
        console.error("WebSocket error:", err);
        ws.close();
      };
    }

    connect();

    return () => {
      if (ws) {
        ws.onclose = null;
        ws.close();
      }
      clearTimeout(reconnectTimer);
    };
  }, [url]);

  const sendMessage = (text: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      addMessage({
        id: Date.now().toString(),
        role: 'user',
        content: text,
        timestamp: new Date(),
        isStreaming: false
      });
      wsRef.current.send(text);
      setStreaming(true);
    }
  };

  return { sendMessage };
}
