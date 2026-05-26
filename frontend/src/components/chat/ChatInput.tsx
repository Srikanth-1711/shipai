import React, { useState } from 'react';
import { useShipAIStore } from '../../store/useShipAIStore';
import { Send, Loader2 } from 'lucide-react';

export const ChatInput: React.FC<{ onSend: (text: string) => void }> = ({ onSend }) => {
  const [text, setText] = useState('');
  const { isStreaming, architectureComplete, requirementsComplete } = useShipAIStore();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim() || isStreaming) return;
    onSend(text);
    setText('');
  };

  let placeholder = "Tell me about your project...";
  if (isStreaming) {
    if (!requirementsComplete) placeholder = "Gathering requirements...";
    else if (!architectureComplete) placeholder = "Analysing books & designing architecture...";
    else placeholder = "Generating your project...";
  } else if (architectureComplete) {
    placeholder = "Ask a follow-up question...";
  }

  return (
    <form onSubmit={handleSubmit} className="p-4 bg-gray-900 flex items-center gap-2">
      <input
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={isStreaming}
        placeholder={placeholder}
        className="flex-1 bg-gray-800 text-white rounded-md px-4 py-3 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:opacity-50"
      />
      <button 
        type="submit" 
        disabled={isStreaming || !text.trim()}
        className="p-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:hover:bg-blue-600 transition-colors"
      >
        {isStreaming ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
      </button>
    </form>
  );
};
