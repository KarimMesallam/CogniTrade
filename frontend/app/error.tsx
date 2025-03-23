'use client';

import { useEffect } from 'react';

interface ErrorProps {
  error: Error;
  reset: () => void;
}

export default function Error({ error, reset }: ErrorProps) {
  useEffect(() => {
    console.error('Application error:', error);
  }, [error]);

  return (
    <div className="flex items-center justify-center min-h-screen bg-slate-900 text-white">
      <div className="text-center p-6 max-w-md">
        <h1 className="text-4xl font-bold mb-4">Something went wrong</h1>
        <p className="mb-6">We've encountered an error while loading this page.</p>
        <div className="flex justify-center space-x-4">
          <button
            onClick={reset}
            className="px-4 py-2 bg-blue-600 rounded hover:bg-blue-700 transition-colors"
          >
            Try again
          </button>
          <a
            href="/"
            className="px-4 py-2 bg-slate-700 rounded hover:bg-slate-600 transition-colors"
          >
            Go to Home
          </a>
        </div>
      </div>
    </div>
  );
} 