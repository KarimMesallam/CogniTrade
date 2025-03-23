'use client';

import { useEffect } from 'react';
import dynamic from 'next/dynamic';
import wsClient from '../lib/ws-client';

// Import WebSocketProvider dynamically with SSR disabled
const WebSocketProvider = dynamic(
  () => import("../lib/websocket-context").then((mod) => mod.WebSocketProvider),
  { ssr: false }
);

interface ClientWrapperProps {
  children: React.ReactNode;
}

export default function ClientWrapper({ children }: ClientWrapperProps) {
  // Initialize WebSocket connection when component mounts
  useEffect(() => {
    if (typeof window !== 'undefined' && wsClient) {
      wsClient.connect();
    }
    
    return () => {
      if (typeof window !== 'undefined' && wsClient) {
        wsClient.disconnect();
      }
    };
  }, []);

  return (
    <WebSocketProvider>
      {children}
    </WebSocketProvider>
  );
} 