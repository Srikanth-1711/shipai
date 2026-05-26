import React from 'react';
import { motion } from 'framer-motion';
import type { Message } from '../../types';
import { Brain, Cpu, Code, BookOpen, User } from 'lucide-react';
import { TypingIndicator } from './TypingIndicator';

const agentStyles = {
  user: {
    border: '',
    bg: 'bg-gray-800',
    icon: <User className="w-5 h-5 text-gray-400" />,
    align: 'justify-end',
    text: 'text-gray-100'
  },
  discovery: {
    border: 'border-l-4 border-agent-discovery',
    bg: 'bg-gray-900',
    icon: <Brain className="w-5 h-5 text-agent-discovery" />,
    align: 'justify-start',
    text: 'text-gray-200'
  },
  architect: {
    border: 'border-l-4 border-agent-architect',
    bg: 'bg-gray-900',
    icon: <Cpu className="w-5 h-5 text-agent-architect" />,
    align: 'justify-start',
    text: 'text-gray-200'
  },
  builder: {
    border: 'border-l-4 border-agent-builder',
    bg: 'bg-gray-900',
    icon: <Code className="w-5 h-5 text-agent-builder" />,
    align: 'justify-start',
    text: 'text-gray-200'
  },
  explainer: {
    border: 'border-l-4 border-agent-explainer',
    bg: 'bg-gray-900',
    icon: <BookOpen className="w-5 h-5 text-agent-explainer" />,
    align: 'justify-start',
    text: 'text-gray-200'
  }
};

export const MessageBubble: React.FC<{ message: Message }> = ({ message }) => {
  const isUser = message.role === 'user';
  const style = agentStyles[message.role] || agentStyles.user;
  
  if (!isUser && message.content === '' && message.isStreaming) {
    return <TypingIndicator />;
  }

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className={`flex w-full mb-4 ${style.align}`}
    >
      <div className={`flex max-w-[85%] rounded-lg p-3 ${style.bg} ${style.border} shadow-sm`}>
        {!isUser && (
          <div className="mr-3 mt-0.5 flex-shrink-0">
            {style.icon}
          </div>
        )}
        <div className={`whitespace-pre-wrap ${style.text} text-sm leading-relaxed`}>
          {message.content}
        </div>
      </div>
    </motion.div>
  );
};
