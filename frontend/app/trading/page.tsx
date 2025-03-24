'use client';

import React, { useEffect, useState } from 'react';
import DashboardLayout from '../../components/DashboardLayout';
import ClientWrapper from '../../components/ClientWrapper';
import dynamic from 'next/dynamic';
import apiService from '../../lib/api-service';
import { useWebSocket } from '../../lib/websocket-context';

// MUI Components
import {
  Typography,
  Box,
  Card,
  CardContent,
  CardHeader,
  Button,
  Select,
  MenuItem,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  FormControl,
  InputLabel,
  SelectChangeEvent,
  CircularProgress
} from '@mui/material';

// Icons
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PowerSettingsNewIcon from '@mui/icons-material/PowerSettingsNew';

// Import TradingChart dynamically with SSR disabled
const TradingChart = dynamic(() => import('../../components/TradingChart'), {
  ssr: false,
});

// Define types for our data
interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface Trade {
  time: number;
  price: number;
  type: 'buy' | 'sell';
}

interface Symbol {
  value: string;
  name: string;
}

interface Signal {
  time: string;
  symbol: string;
  strategy: string;
  signal: string;
  confidence: number;
  price: number;
  action_taken: string;
}

const timeframes = [
  { value: '1m', name: '1 minute' },
  { value: '5m', name: '5 minutes' },
  { value: '15m', name: '15 minutes' },
  { value: '1h', name: '1 hour' },
  { value: '4h', name: '4 hours' },
  { value: '1d', name: '1 day' },
];

export default function TradingPage() {
  // Add a clientSide flag to prevent hydration issues
  const [clientSide, setClientSide] = useState(false);
  
  // Initialize with empty state values to prevent hydration mismatches
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [timeframe, setTimeframe] = useState('1h');
  const [strategy, setStrategy] = useState('sma_crossover');
  const [isBotRunning, setIsBotRunning] = useState(false);
  const [candleData, setCandleData] = useState<Candle[]>([]);
  const [tradeData, setTradeData] = useState<Trade[]>([]);
  const [symbols, setSymbols] = useState<Symbol[]>([]);
  const [strategies, setStrategies] = useState<{value: string, name: string}[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isMounted, setIsMounted] = useState(false);
  
  const { isConnected, lastMessage, messages } = useWebSocket();

  // Set clientSide flag on mount - this happens before any other effects
  useEffect(() => {
    setClientSide(true);
  }, []);

  // Handle component mounting to prevent hydration errors
  useEffect(() => {
    setIsMounted(true);
  }, []);

  // Handle WebSocket messages for real-time updates
  useEffect(() => {
    if (!isMounted) return;
    
    if (lastMessage) {
      if (lastMessage.type === 'trading_status') {
        setIsBotRunning(lastMessage.status === 'started');
      }
      else if (lastMessage.type === 'signal') {
        // Add the new signal to the signals array
        setSignals(prev => [lastMessage.data, ...prev].slice(0, 10));
      }
    }
  }, [lastMessage, isMounted]);

  // Fetch market data when the symbol or timeframe changes
  useEffect(() => {
    if (!isMounted) return;
    
    async function fetchMarketData() {
      try {
        const response = await apiService.getMarketData(symbol, timeframe);
        if (response.status === 'success' && response.candles) {
          const formattedCandles = response.candles
            .map((candle: any) => ({
              time: candle.time, // Keep as timestamp for sorting
              open: candle.open,
              high: candle.high,
              low: candle.low,
              close: candle.close,
              volume: candle.volume
            }))
            // Sort candles by timestamp in ascending order
            .sort((a: any, b: any) => a.time - b.time);
            
          // Remove any duplicate timestamps
          const uniqueCandles = formattedCandles.reduce((acc: any[], current: any) => {
            const x = acc.find((item: any) => item.time === current.time);
            if (!x) {
              return acc.concat([current]);
            } else {
              // If there are duplicates, keep the latest one with most recent data
              return acc;
            }
          }, []);
            
          setCandleData(uniqueCandles);
        }
      } catch (error) {
        console.error('Failed to fetch market data:', error);
      }
    }

    fetchMarketData();
  }, [symbol, timeframe, isMounted]);

  // Fetch signals when the symbol changes
  useEffect(() => {
    if (!isMounted) return;
    
    async function fetchSignals() {
      try {
        const response = await apiService.getSignalHistory(symbol, 10);
        // Check if response has status field and it's 'success'
        if (response && response.status === 'success' && Array.isArray(response.signals)) {
          setSignals(response.signals);
        } else if (response && response.status === 'error') {
          console.warn('Error fetching signals:', response.message);
          // Set empty signals instead of throwing
          setSignals([]);
        }
      } catch (error) {
        console.error('Failed to fetch signals:', error);
        // Set empty signals array on error
        setSignals([]);
      }
    }

    fetchSignals();
  }, [symbol, isMounted]);

  // Fetch symbols and strategies when the component mounts
  useEffect(() => {
    if (!isMounted) return;
    
    async function fetchInitialData() {
      setIsLoading(true);
      try {
        // Fetch symbols
        const symbolsResponse = await apiService.getSymbols();
        if (symbolsResponse && symbolsResponse.status === 'success' && Array.isArray(symbolsResponse.symbols)) {
          const formattedSymbols = symbolsResponse.symbols.map((s: any) => ({
            value: s.symbol,
            name: `${s.baseAsset}/${s.quoteAsset}`
          }));
          setSymbols(formattedSymbols);
        } else {
          // Default to empty array if API call fails
          setSymbols([]);
          console.warn('Could not load symbols:', 
            symbolsResponse?.message || 'Unknown error');
        }

        // Fetch strategies
        const strategiesResponse = await apiService.getStrategies();
        if (strategiesResponse && strategiesResponse.status === 'success' && Array.isArray(strategiesResponse.strategies)) {
          const formattedStrategies = strategiesResponse.strategies.map((s: any) => ({
            value: s.name,
            name: s.description
          }));
          setStrategies(formattedStrategies);
        } else {
          // Default to empty array if API call fails
          setStrategies([]);
          console.warn('Could not load strategies:', 
            strategiesResponse?.message || 'Unknown error');
        }
      } catch (error) {
        console.error('Failed to fetch initial data:', error);
        // Set defaults on error
        setSymbols([]);
        setStrategies([]);
      } finally {
        setIsLoading(false);
      }
    }

    fetchInitialData();
  }, [isMounted]);

  const toggleBot = async () => {
    try {
      if (isBotRunning) {
        const response = await apiService.stopTrading();
        if (response.status === 'success') {
          setIsBotRunning(false);
        }
      } else {
        const response = await apiService.startTrading({
          symbol,
          interval: timeframe,
          trade_amount: 100,
          strategies: [
            {
              name: strategy,
              params: {},
              active: true
            }
          ]
        });
        if (response.status === 'success') {
          setIsBotRunning(true);
        }
      }
    } catch (error) {
      console.error('Failed to toggle bot:', error);
    }
  };

  // Conditionally render the entire page based on clientSide flag
  if (!clientSide) {
    return (
      <ClientWrapper>
        <DashboardLayout>
          <Box 
            sx={{ 
              display: 'flex', 
              justifyContent: 'center', 
              alignItems: 'center', 
              height: '80vh' 
            }}
          >
            <CircularProgress />
          </Box>
        </DashboardLayout>
      </ClientWrapper>
    );
  }

  return (
    <ClientWrapper>
      <DashboardLayout>
        {isMounted ? (
          <>
            <Card sx={{ mb: 3, bgcolor: 'background.paper', borderRadius: 1 }}>
              <CardContent>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={4}>
                    <Typography variant="h5" component="h1" gutterBottom>
                      Live Trading
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Configure and manage your trading bot
                    </Typography>
                  </Grid>

                  <Grid item xs={12} md={8}>
                    <Grid container spacing={2} alignItems="center">
                      <Grid item xs={12} sm={6} md={3}>
                        <FormControl fullWidth size="small">
                          <InputLabel id="symbol-select-label">Symbol</InputLabel>
                          <Select
                            labelId="symbol-select-label"
                            value={symbol}
                            label="Symbol"
                            onChange={(e: SelectChangeEvent) => setSymbol(e.target.value)}
                            disabled={isLoading}
                          >
                            {symbols.map((s) => (
                              <MenuItem key={s.value} value={s.value}>
                                {s.name}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      </Grid>

                      <Grid item xs={12} sm={6} md={3}>
                        <FormControl fullWidth size="small">
                          <InputLabel id="timeframe-select-label">Timeframe</InputLabel>
                          <Select
                            labelId="timeframe-select-label"
                            value={timeframe}
                            label="Timeframe"
                            onChange={(e: SelectChangeEvent) => setTimeframe(e.target.value)}
                          >
                            {timeframes.map((t) => (
                              <MenuItem key={t.value} value={t.value}>
                                {t.name}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      </Grid>

                      <Grid item xs={12} sm={6} md={3}>
                        <FormControl fullWidth size="small">
                          <InputLabel id="strategy-select-label">Strategy</InputLabel>
                          <Select
                            labelId="strategy-select-label"
                            value={strategy}
                            label="Strategy"
                            onChange={(e: SelectChangeEvent) => setStrategy(e.target.value)}
                            disabled={isLoading}
                          >
                            {strategies.map((s) => (
                              <MenuItem key={s.value} value={s.value}>
                                {s.name}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      </Grid>

                      <Grid item xs={12} sm={6} md={3}>
                        <Button
                          fullWidth
                          variant="contained"
                          color={isBotRunning ? "error" : "success"}
                          onClick={toggleBot}
                          startIcon={isBotRunning ? <PowerSettingsNewIcon /> : <PlayArrowIcon />}
                          disabled={isLoading}
                        >
                          {isBotRunning ? 'Stop Bot' : 'Start Bot'}
                        </Button>
                      </Grid>
                    </Grid>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>

            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Card sx={{ bgcolor: 'background.paper' }}>
                  <CardContent>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="h6">{symbol}</Typography>
                        <Typography variant="body2" color="text.secondary">{timeframe}</Typography>
                      </Box>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Box 
                          sx={{ 
                            height: 10, 
                            width: 10, 
                            borderRadius: '50%', 
                            bgcolor: isConnected ? (isBotRunning ? 'success.main' : 'warning.main') : 'error.main'
                          }}
                        />
                        <Typography variant="body2">
                          {isConnected ? (isBotRunning ? 'Bot Running' : 'Bot Ready') : 'Disconnected'}
                        </Typography>
                      </Box>
                    </Box>

                    <Box sx={{ height: 400 }}>
                      {candleData.length > 0 ? (
                        <TradingChart candles={candleData} trades={tradeData} height={400} />
                      ) : (
                        <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <Typography>Loading chart data...</Typography>
                        </Box>
                      )}
                    </Box>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Card sx={{ bgcolor: 'background.paper' }}>
                  <CardHeader 
                    title="Recent Signals" 
                    sx={{ bgcolor: 'action.selected', borderBottom: 1, borderColor: 'divider' }}
                  />
                  
                  <CardContent>
                    <TableContainer sx={{ maxHeight: 400 }}>
                      <Table stickyHeader>
                        <TableHead>
                          <TableRow>
                            <TableCell>Time</TableCell>
                            <TableCell>Symbol</TableCell>
                            <TableCell>Strategy</TableCell>
                            <TableCell>Signal</TableCell>
                            <TableCell>Strength</TableCell>
                            <TableCell>Action Taken</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {signals.length > 0 ? (
                            signals.map((signal, index) => (
                              <TableRow key={index} hover>
                                <TableCell>{signal.time}</TableCell>
                                <TableCell>{signal.symbol}</TableCell>
                                <TableCell>{signal.strategy}</TableCell>
                                <TableCell sx={{ color: signal.signal === 'BUY' ? 'success.main' : 'error.main', fontWeight: 'medium' }}>
                                  {signal.signal}
                                </TableCell>
                                <TableCell>
                                  {typeof signal.confidence === 'number' ? signal.confidence.toFixed(2) : signal.confidence}
                                </TableCell>
                                <TableCell sx={{ color: signal.action_taken.includes('Order') ? 'success.main' : 'text.secondary' }}>
                                  {signal.action_taken}
                                </TableCell>
                              </TableRow>
                            ))
                          ) : (
                            <TableRow>
                              <TableCell colSpan={6} align="center">
                                No signals available
                              </TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </>
        ) : (
          <Box sx={{ p: 4, textAlign: 'center' }}>
            <Typography>Loading trading interface...</Typography>
          </Box>
        )}
      </DashboardLayout>
    </ClientWrapper>
  );
} 