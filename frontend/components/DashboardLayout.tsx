'use client';

import React from 'react';
import Navbar from './Navbar';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <div className="flex h-screen bg-slate-900">
      <Navbar />
      <main className="flex-1 pl-64 overflow-auto bg-slate-900 text-white">
        <div className="p-6">
          {children}
        </div>
      </main>
    </div>
  );
} 