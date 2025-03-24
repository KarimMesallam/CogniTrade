'use client';

import React from 'react';
import Navbar from './Navbar';
import { Box } from '@mui/material';

interface DashboardLayoutProps {
  children: React.ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <Box sx={{ display: 'flex', minHeight: '100vh', bgcolor: 'background.default' }}>
      <Navbar />
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          pl: { xs: 0, sm: 8 },
          pt: 0,
          width: { sm: `calc(100% - 240px)` },
          overflow: 'auto',
          color: 'text.primary'
        }}
      >
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      </Box>
    </Box>
  );
} 