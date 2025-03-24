'use client';

import React, { useEffect, useState, Suspense } from 'react';
import dynamic from 'next/dynamic';

// Import WebSocketProvider directly instead of using an intermediate component
const WebSocketProvider = dynamic(
  () => import("../lib/websocket-context").then(mod => mod.WebSocketProvider),
  { ssr: false }
);

interface ClientWrapperProps {
  children: React.ReactNode;
}

export default function ClientWrapper({ children }: ClientWrapperProps) {
  const [isClientMounted, setIsClientMounted] = useState(false);

  // Handle client-side mounting first
  useEffect(() => {
    setIsClientMounted(true);
  }, []);

  // Don't render any WebSocket related components during SSR
  if (!isClientMounted) {
    return <>{children}</>;
  }

  // Only render the WebSocketProvider on the client
  return (
    <Suspense fallback={<>{children}</>}>
      <WebSocketProvider>
        {children}
      </WebSocketProvider>
    </Suspense>
  );
} 