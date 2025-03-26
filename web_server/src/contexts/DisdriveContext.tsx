import { createContext, useState, useContext } from "react";

interface DisdriveContextType {
  isLogging: boolean;
  setIsLogging: (value: boolean) => void;
  hasOngoingSession: boolean;
  setHasOngoingSession: (value: boolean) => void;
}

const DisdriveContext = createContext<DisdriveContextType | undefined>(
  undefined
); // or define a type for better safety

export const DisdriveProvider = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const [isLogging, setIsLogging] = useState(true);
  const [hasOngoingSession, setHasOngoingSession] = useState(true);

  return (
    <DisdriveContext.Provider
      value={{
        isLogging,
        setIsLogging,
        hasOngoingSession,
        setHasOngoingSession,
      }}
    >
      {children}
    </DisdriveContext.Provider>
  );
};

// Custom hook (optional but recommended)
export const useDisdriveContext = () => {
  const context = useContext(DisdriveContext);
  if (!context) {
    throw new Error(
      "DisdriveContext must be used within a DisdriveContextProvider"
    );
  }
  return context;
};
