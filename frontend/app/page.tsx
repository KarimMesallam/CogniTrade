'use client';

import { Typography, Box, Button, Stack } from '@mui/material';

// Dashboard client component to prevent server errors
export default function Home() {
  return (
    <Box sx={{ 
      p: 8, 
      bgcolor: 'background.default', 
      minHeight: '100vh', 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      justifyContent: 'center'
    }}>
      <Box sx={{ maxWidth: '600px', textAlign: 'center' }}>
        <Typography variant="h2" fontWeight="bold" gutterBottom>
          AI Trading Bot Dashboard
        </Typography>
        <Typography variant="h5" sx={{ mb: 4, color: 'text.secondary' }}>
          Web UI for AI Trading Bot with Binance & LLM Integration
        </Typography>
        
        <Stack direction="row" spacing={2} justifyContent="center">
          <Button 
            variant="contained" 
            color="primary" 
            component="a" 
            href="/dashboard" 
            sx={{ px: 3, py: 1.5 }}
          >
            Dashboard
          </Button>
          <Button 
            variant="outlined" 
            component="a" 
            href="/trading" 
            sx={{ px: 3, py: 1.5 }}
          >
            Live Trading
          </Button>
          <Button 
            variant="outlined"
            component="a" 
            href="/backtesting" 
            sx={{ px: 3, py: 1.5 }}
          >
            Backtesting
          </Button>
        </Stack>
      </Box>
    </Box>
  );
}
