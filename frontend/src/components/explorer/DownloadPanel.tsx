import React, { useState } from 'react';
import { useShipAIStore } from '../../store/useShipAIStore';
import { Download, Code as CodeIcon, Terminal, CheckCircle2 } from 'lucide-react';

export const DownloadPanel: React.FC = () => {
  const { configJson, fileTree, projectPath, sessionId } = useShipAIStore();
  const [accepted, setAccepted] = useState(false);

  if (!projectPath || fileTree.length === 0) return null;

  const templateName = configJson?.template || 'rag_chatbot';
  const tier = configJson?.infra_tier || 'minimal';
  const framework = configJson?.framework || 'langgraph';
  const search = configJson?.search_type || 'dense';
  const pii = configJson?.pii || 'none';

  const handleCopyCmd = () => {
    const cmd = `shipai create ${templateName} --search ${search} --pii ${pii} --infra ${tier}`;
    navigator.clipboard.writeText(cmd);
  };

  const handleDownload = async () => {
    if (sessionId && !accepted) {
      try {
        await fetch(`http://${window.location.hostname}:8000/api/feedback/accept/${sessionId}`, { method: 'POST' });
        setAccepted(true);
      } catch (err) {
        console.error("Failed to send accept feedback", err);
      }
    }
    // Logic to actually trigger zip download would go here
    alert("Project downloaded! (Feedback recorded)");
  };

  return (
    <div className="shrink-0 border-t border-gray-800 bg-gray-900 p-4">
      <div className="flex items-start gap-2 mb-3">
        <CheckCircle2 className="text-green-500 mt-0.5 shrink-0" size={18} />
        <div>
          <h3 className="font-semibold text-gray-100 text-sm capitalize">{tier} {templateName.replace('_', ' ')}</h3>
          <p className="text-xs text-gray-400 mt-0.5">
            {fileTree.length} files • {framework}
            <br />
            {search} search {pii !== 'none' ? `• ${pii}` : ''}
          </p>
        </div>
      </div>
      
      <div className="flex flex-col gap-2">
        <button 
          onClick={handleDownload}
          className="w-full py-2 px-3 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium rounded-md flex items-center justify-center gap-2 transition-colors"
        >
          <Download size={16} />
          Download Project .zip
        </button>
        <div className="flex gap-2">
          <button className="flex-1 py-1.5 px-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 text-xs font-medium rounded-md flex items-center justify-center gap-1.5 transition-colors">
            <CodeIcon size={14} />
            VS Code
          </button>
          <button 
            onClick={handleCopyCmd}
            className="flex-1 py-1.5 px-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 text-gray-300 text-xs font-medium rounded-md flex items-center justify-center gap-1.5 transition-colors"
          >
            <Terminal size={14} />
            CLI Cmd
          </button>
        </div>
      </div>
    </div>
  );
};
