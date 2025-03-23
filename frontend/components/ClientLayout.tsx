'use client';

import { WebSocketProvider } from "../lib/websocket-context";

export function ClientLayout({ 
  children 
}: { 
  children: React.ReactNode 
}) {
  return (
    <WebSocketProvider>
      {children}
    </WebSocketProvider>
  );
} 