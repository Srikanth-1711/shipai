import { useShipAIStore } from '../../store/useShipAIStore';
import { FileCode, Folder, Star } from 'lucide-react';

export const FileTree = () => {
  const { fileTree, selectedFile, setSelectedFile } = useShipAIStore();

  if (fileTree.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-500 text-sm">
        Files will appear here when generation starts...
      </div>
    );
  }

  // Simplified flat list render for now (could be recursive for nested tree)
  return (
    <div className="flex-1 overflow-y-auto p-2 custom-scrollbar">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 px-2">Generated Project</div>
      {fileTree.map((file, idx) => {
        const isSelected = selectedFile === file.path;
        return (
          <button
            key={idx}
            onClick={() => setSelectedFile(file.path)}
            className={`w-full text-left flex items-center gap-2 px-2 py-1.5 rounded-md text-sm transition-colors ${
              isSelected ? 'bg-gray-800 text-blue-400' : 'text-gray-400 hover:bg-gray-800/50 hover:text-gray-200'
            }`}
          >
            {file.type === 'directory' ? <Folder size={16} /> : <FileCode size={16} />}
            <span className="truncate flex-1">{file.name}</span>
            {file.highlight && <Star size={14} className="text-amber-400 shrink-0" fill="currentColor" />}
          </button>
        );
      })}
    </div>
  );
};
