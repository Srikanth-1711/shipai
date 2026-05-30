import React from 'react';
import { useShipAIStore } from '../../store/useShipAIStore';
import { FileTree } from './FileTree';
import { FileViewer } from './FileViewer';
import { DownloadPanel } from './DownloadPanel';
import { ConfigEditor } from './ConfigEditor';

export const CodeExplorer: React.FC = () => {
  const { architectureComplete } = useShipAIStore();

  return (
    <div className="w-full h-full bg-gray-950 flex flex-col border-l border-gray-800">
      <div className="p-4 border-b border-gray-800 bg-gray-900 shrink-0">
        <h2 className="text-lg font-bold text-gray-100">Code Explorer</h2>
      </div>
      
      <div className="flex-1 flex flex-col overflow-hidden relative">
        {!architectureComplete ? (
          <div className="absolute inset-0 flex items-center justify-center text-gray-500 text-sm z-10 bg-gray-950/80 backdrop-blur-sm">
            Architecture not finalized yet...
          </div>
        ) : null}
        
        <div className="flex-1 flex flex-col min-h-0">
          <div className="h-1/3 min-h-[150px] border-b border-gray-800 flex flex-col overflow-hidden">
            <FileTree />
          </div>
          <div className="flex-1 flex flex-col overflow-hidden">
            <FileViewer />
          </div>
        </div>
      </div>
      
      <ConfigEditor />
      <DownloadPanel />
    </div>
  );
};
