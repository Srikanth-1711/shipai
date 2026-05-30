import { create } from 'zustand';
import type { Message, AgentType, SystemRequirements, CanvasNode, CanvasEdge, FileNode } from '../types';

interface ShipAIState {
  // Chat
  messages: Message[];
  isStreaming: boolean;
  currentAgent: AgentType;
  
  // Requirements progress
  requirements: Partial<SystemRequirements>;
  requirementsComplete: boolean;
  
  // Canvas
  canvasNodes: CanvasNode[];
  canvasEdges: CanvasEdge[];
  architectureComplete: boolean;
  
  // Code explorer
  fileTree: FileNode[];
  selectedFile: string | null;
  projectPath: string | null;
  configJson: Record<string, any>;
  sessionId: string | null;
  
  // Actions
  addMessage: (msg: Message) => void;
  appendToken: (token: string) => void;
  setStreaming: (isStreaming: boolean) => void;
  setCurrentAgent: (agent: AgentType) => void;
  updateRequirement: (field: string, value: string) => void;
  addCanvasNode: (node: CanvasNode) => void;
  updateCanvasNode: (id: string, updates: Partial<CanvasNode>) => void;
  addCanvasEdge: (edge: CanvasEdge) => void;
  setFileTree: (tree: FileNode[]) => void;
  setProjectData: (path: string, config: any, sessionId: string) => void;
  setSelectedFile: (path: string | null) => void;
  setRequirementsComplete: (complete: boolean) => void;
  setArchitectureComplete: (complete: boolean) => void;
}

export const useShipAIStore = create<ShipAIState>((set) => ({
  messages: [],
  isStreaming: false,
  currentAgent: 'user',
  
  requirements: {},
  requirementsComplete: false,
  
  canvasNodes: [],
  canvasEdges: [],
  architectureComplete: false,
  
  fileTree: [],
  selectedFile: null,
  projectPath: null,
  configJson: {},
  sessionId: null,
  
  addMessage: (msg) => set((state) => ({ messages: [...state.messages, msg] })),
  
  appendToken: (token) => set((state) => {
    const messages = [...state.messages];
    if (messages.length === 0) return state;
    const lastMessage = messages[messages.length - 1];
    
    // Only append if the last message is from the assistant/agent
    if (lastMessage.role !== 'user') {
      lastMessage.content += token;
    }
    return { messages };
  }),
  
  setStreaming: (isStreaming) => set({ isStreaming }),
  
  setCurrentAgent: (currentAgent) => set({ currentAgent }),
  
  updateRequirement: (field, value) => set((state) => ({
    requirements: { ...state.requirements, [field]: value }
  })),
  
  addCanvasNode: (node) => set((state) => {
    if (state.canvasNodes.find(n => n.id === node.id)) return state;
    return { canvasNodes: [...state.canvasNodes, node] };
  }),
  
  updateCanvasNode: (id, updates) => set((state) => ({
    canvasNodes: state.canvasNodes.map(n => n.id === id ? { ...n, ...updates } : n)
  })),
  
  addCanvasEdge: (edge) => set((state) => {
    if (state.canvasEdges.find(e => e.id === edge.id)) return state;
    return { canvasEdges: [...state.canvasEdges, edge] };
  }),
  
  setFileTree: (fileTree) => set({ fileTree }),
  
  setProjectData: (projectPath, configJson, sessionId) => set({ projectPath, configJson, sessionId }),
  
  setSelectedFile: (selectedFile) => set({ selectedFile }),
  
  setRequirementsComplete: (requirementsComplete) => set({ requirementsComplete }),
  
  setArchitectureComplete: (architectureComplete) => set({ architectureComplete })
}));
