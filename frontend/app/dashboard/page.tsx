'use client';

import React, { useState } from 'react';
import DashboardLayout from '../../components/DashboardLayout';
import ClientWrapper from '../../components/ClientWrapper';
import { 
  Typography, 
  Card, 
  CardContent, 
  CardHeader, 
  Grid, 
  Box, 
  Table, 
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper
} from '@mui/material';
import { LineChart } from '@mui/x-charts/LineChart';
import { BarChart } from '@mui/x-charts/BarChart';

// Placeholder data for demonstration
const performanceData = [
  { date: new Date(2022, 0, 22), profit: 2890 },
  { date: new Date(2022, 1, 22), profit: 1890 },
  { date: new Date(2022, 2, 22), profit: 3890 },
  { date: new Date(2022, 3, 22), profit: -1290 },
  { date: new Date(2022, 4, 22), profit: 1890 },
  { date: new Date(2022, 5, 22), profit: 2890 },
];

const strategyPerformance = [
  { name: 'Moving Average Crossover', value: 421 },
  { name: 'RSI Strategy', value: 294 },
  { name: 'Bollinger Bands', value: -308 },
  { name: 'MACD Strategy', value: 152 },
];

export default function Dashboard() {
  const [activeStrategy, setActiveStrategy] = useState('Moving Average Crossover');

  return (
    <ClientWrapper>
      <DashboardLayout>
        <Typography variant="h4" gutterBottom>Trading Dashboard</Typography>

        <Grid container spacing={3} mb={3}>
          <Grid item xs={12} md={4}>
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardContent>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Total Profit/Loss
                </Typography>
                <Typography variant="h3">
                  $2,450.56
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardContent>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Win Rate
                </Typography>
                <Typography variant="h3">
                  68.2%
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardContent>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Total Trades
                </Typography>
                <Typography variant="h3">
                  42
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        <Grid container spacing={3} mb={3}>
          <Grid item xs={12} lg={6}>
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardHeader title="Monthly Performance" />
              <CardContent>
                <Box sx={{ height: 300 }}>
                  <LineChart
                    xAxis={[{
                      data: performanceData.map(item => item.date),
                      scaleType: 'time',
                    }]}
                    series={[{
                      data: performanceData.map(item => item.profit),
                      label: 'Profit/Loss',
                      color: '#2196f3',
                      area: true,
                    }]}
                    height={300}
                    margin={{ top: 20, right: 20, bottom: 30, left: 40 }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} lg={6}>
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardHeader 
                title="Strategy Performance" 
                subheader="Profit/Loss by strategy ($)"
              />
              <CardContent>
                <Box sx={{ height: 300 }}>
                  <BarChart
                    xAxis={[{
                      scaleType: 'band',
                      data: strategyPerformance.map(item => item.name),
                    }]}
                    series={[{
                      data: strategyPerformance.map(item => item.value),
                      color: '#2196f3',
                    }]}
                    height={300}
                    margin={{ top: 20, right: 20, bottom: 50, left: 40 }}
                  />
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardHeader title="Recent Trades" />
              <CardContent>
                <TableContainer component={Paper} sx={{ maxHeight: 'calc(100vh - 500px)', overflow: 'auto' }}>
                  <Table stickyHeader>
                    <TableHead>
                      <TableRow>
                        <TableCell>Time</TableCell>
                        <TableCell>Symbol</TableCell>
                        <TableCell>Type</TableCell>
                        <TableCell>Price</TableCell>
                        <TableCell>Quantity</TableCell>
                        <TableCell>P/L</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      <TableRow>
                        <TableCell>2023-03-23 14:22:05</TableCell>
                        <TableCell>BTCUSDT</TableCell>
                        <TableCell sx={{ color: 'success.main' }}>BUY</TableCell>
                        <TableCell>$42,560.78</TableCell>
                        <TableCell>0.25</TableCell>
                        <TableCell sx={{ color: 'success.main' }}>+$125.45</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>2023-03-23 11:45:32</TableCell>
                        <TableCell>ETHUSDT</TableCell>
                        <TableCell sx={{ color: 'error.main' }}>SELL</TableCell>
                        <TableCell>$2,785.25</TableCell>
                        <TableCell>1.5</TableCell>
                        <TableCell sx={{ color: 'error.main' }}>-$42.32</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell>2023-03-22 18:33:17</TableCell>
                        <TableCell>BTCUSDT</TableCell>
                        <TableCell sx={{ color: 'success.main' }}>BUY</TableCell>
                        <TableCell>$42,105.92</TableCell>
                        <TableCell>0.15</TableCell>
                        <TableCell sx={{ color: 'success.main' }}>+$87.21</TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </DashboardLayout>
    </ClientWrapper>
  );
} 