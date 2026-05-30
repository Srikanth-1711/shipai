export type AgentType = "user" | "discovery" | "architect" | "builder" | "explainer";

export interface Message {
  id: string;
  role: AgentType;
  content: string;
  timestamp: Date;
  isStreaming: boolean;
}

export interface SystemRequirements {
  user_level: string;
  use_case: string;
  data_type: string;
  data_location: string;
  data_change_rate: string;
  scale: string;
  query_type: string;
  privacy_level: string;
}

export interface CanvasNode {
  id: string;
  type: string; // "OrchestratorNode", "VectorDBNode", etc.
  label: string;
  status: "pending" | "deciding" | "confirmed";
  x?: number;
  y?: number;
}

export interface CanvasEdge {
  id: string;
  source: string;
  target: string;
  type: "data flow" | "async/queue" | "monitoring";
}

export interface FileNode {
  name: string;
  type: "file" | "directory";
  path: string;
  children?: FileNode[];
  highlight?: boolean;
}

export interface WSEvent {
  type: "token" | "agent_change" | "requirement" | "decision" | "file_generated" | "complete" | "error";
  payload: any;
}
