import React, { useEffect, useRef } from 'react';
import { useShipAIStore } from '../../store/useShipAIStore';
import { MessageBubble } from './MessageBubble';
import { RequirementsBar } from './RequirementsBar';
import { ChatInput } from './ChatInput';

export const ChatPanel: React.FC<{ onSendMessage: (text: string) => void }> = ({ onSendMessage }) => {
  const { messages } = useShipAIStore();
  const endOfMessagesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="flex flex-col h-full bg-gray-950 border-r border-gray-800">
      <div className="p-4 border-b border-gray-800 bg-gray-900">
        <h2 className="text-lg font-bold text-gray-100 flex items-center gap-2">
          ShipAI
          <span className="text-xs bg-green-900 text-green-400 px-2 py-0.5 rounded-full border border-green-700">status: ready</span>
        </h2>
      </div>
      
      <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
        {messages.length === 0 ? (
          <div className="h-full flex items-center justify-center text-gray-500 italic">
            Start by telling ShipAI about your project...
          </div>
        ) : (
          messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))
        )}
        <div ref={endOfMessagesRef} />
      </div>

      <div className="shrink-0">
        <RequirementsBar />
        <ChatInput onSend={onSendMessage} />
      </div>
    </div>
  );
};
