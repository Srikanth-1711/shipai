import { useWebSocket } from './hooks/useWebSocket';
import { ChatPanel } from './components/chat/ChatPanel';
import { ArchitectureCanvas } from './components/canvas/ArchitectureCanvas';
import { CodeExplorer } from './components/explorer/CodeExplorer';

function App() {
  const { sendMessage } = useWebSocket(`ws://${window.location.hostname}:8000/ws/chat`);

  return (
    <div className="flex h-screen w-full bg-gray-950 overflow-hidden font-sans text-gray-100">
      {/* Three Panel Layout Container */}
      <div className="w-[35%] h-full">
        <ChatPanel onSendMessage={sendMessage} />
      </div>
      
      <div className="w-[40%] h-full">
        <ArchitectureCanvas />
      </div>
      
      <div className="w-[25%] h-full">
        <CodeExplorer />
      </div>
    </div>
  );
}

export default App;
