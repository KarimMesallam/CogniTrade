'use client';

import React, { useState } from 'react';
import { Box, Typography, Grid, Button } from '@mui/material';
import { LocalizationProvider, DatePicker } from '@mui/x-date-pickers';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';

export default function DatePickerTestPage() {
  const [startDate, setStartDate] = useState<Date | null>(
    new Date(new Date().getFullYear(), new Date().getMonth() - 3, 1)
  );
  const [endDate, setEndDate] = useState<Date | null>(new Date());

  return (
    <Box sx={{ p: 4 }}>
      <Typography variant="h4" gutterBottom>Date Picker Test</Typography>
      
      <LocalizationProvider dateAdapter={AdapterDateFns}>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <DatePicker
              label="Start Date"
              value={startDate}
              onChange={(newValue) => setStartDate(newValue)}
              slotProps={{ textField: { fullWidth: true } }}
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <DatePicker
              label="End Date"
              value={endDate}
              onChange={(newValue) => setEndDate(newValue)}
              slotProps={{ textField: { fullWidth: true } }}
            />
          </Grid>
        </Grid>
      </LocalizationProvider>
      
      <Box sx={{ mt: 4 }}>
        <Typography variant="h6">Selected Dates:</Typography>
        <Typography>
          Start: {startDate?.toLocaleDateString() || 'None'}
        </Typography>
        <Typography>
          End: {endDate?.toLocaleDateString() || 'None'}
        </Typography>
      </Box>
    </Box>
  );
} 