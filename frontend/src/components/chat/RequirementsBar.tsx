import { useShipAIStore } from '../../store/useShipAIStore';

export const RequirementsBar = () => {
  const { requirements, requirementsComplete } = useShipAIStore();
  
  const requiredFields = [
    'user_level', 'use_case', 'data_type', 'data_location', 
    'data_change_rate', 'scale', 'query_type', 'privacy_level'
  ];
  
  const filledCount = requiredFields.filter(f => requirements[f as keyof typeof requirements]).length;
  const total = requiredFields.length;
  
  const isComplete = filledCount === total || requirementsComplete;
  
  return (
    <div className="w-full px-4 py-2 bg-gray-900 border-t border-gray-800">
      <div className="flex justify-between items-center mb-1 text-xs text-gray-400">
        <span>{isComplete ? 'Requirements Gathered' : `Requirements: ${filledCount}/${total} fields confirmed`}</span>
        {isComplete && <span className="text-green-400 font-semibold">Researching your architecture...</span>}
      </div>
      <div className="w-full bg-gray-800 rounded-full h-1.5 flex overflow-hidden">
        {requiredFields.map((field) => {
          const isFilled = !!requirements[field as keyof typeof requirements] || requirementsComplete;
          return (
            <div 
              key={field} 
              className={`h-full flex-1 border-r border-gray-900 last:border-0 transition-colors duration-500 ${isFilled ? 'bg-green-500' : 'bg-gray-700'}`}
              title={field}
            />
          );
        })}
      </div>
    </div>
  );
};
