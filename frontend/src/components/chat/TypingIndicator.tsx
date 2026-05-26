import { motion } from 'framer-motion';

export const TypingIndicator = () => {
  return (
    <div className="flex space-x-1 items-center p-2 rounded-lg bg-gray-800 w-16 h-8 justify-center ml-2 border-l-2 border-blue-500">
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="w-2 h-2 bg-gray-400 rounded-full"
          animate={{
            y: ["0%", "-50%", "0%"]
          }}
          transition={{
            duration: 0.6,
            repeat: Infinity,
            delay: i * 0.15
          }}
        />
      ))}
    </div>
  );
};
