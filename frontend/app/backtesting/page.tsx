'use client';

import { useState, useEffect } from 'react';
import DashboardLayout from '../../components/DashboardLayout';
import ClientWrapper from '../../components/ClientWrapper';
import dynamic from 'next/dynamic';
import apiService from '../../lib/api-service';

// MUI Components
import { 
  Typography, 
  Card, 
  CardContent, 
  CardHeader, 
  Button, 
  Select, 
  MenuItem,
  InputLabel,
  FormControl,
  TextField,
  Box,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Divider,
  Chip,
  FormGroup,
  ToggleButton,
  ToggleButtonGroup,
  SelectChangeEvent,
  Grid
} from '@mui/material';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { LocalizationProvider, DatePicker } from '@mui/x-date-pickers';
import { LineChart } from '@mui/x-charts/LineChart';

// Icons
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import ShowChartIcon from '@mui/icons-material/ShowChart';
import CalendarTodayIcon from '@mui/icons-material/CalendarToday';

// Import TradingChart dynamically with SSR disabled
const TradingChartDynamic = dynamic(() => import('../../components/TradingChart'), {
  ssr: false,
});

// Define interfaces for data types
interface Strategy {
  id: string;
  name: string;
  description: string;
  parameters: Record<string, any>;
}

interface Symbol {
  value: string;
  name: string;
}

interface BacktestResult {
  metrics: {
    total_trades: number;
    winning_trades: number;
    win_rate: number;
    total_profit: number;
    profit_percent: number;
    max_drawdown: number;
    max_drawdown_percent: number;
    sharpe_ratio: number;
  };
  equity_curve: {
    date: string;
    balance: number;
  }[];
  trades: any[];
}

// TabPanel component for handling tab content
function TabPanel(props: {
  children?: React.ReactNode;
  index: number;
  value: number;
}) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ pt: 2 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

export default function BacktestingPage() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [selectedTimeframes, setSelectedTimeframes] = useState(['1h']);
  const [selectedStrategy, setSelectedStrategy] = useState('');
  const [initialCapital, setInitialCapital] = useState(10000);
  const [commission, setCommission] = useState(0.1);
  const [startDate, setStartDate] = useState<Date | null>(
    new Date(new Date().getFullYear(), new Date().getMonth() - 3, 1) // 3 months ago
  );
  const [endDate, setEndDate] = useState<Date | null>(new Date());
  const [isRunning, setIsRunning] = useState(false);
  const [hasResults, setHasResults] = useState(false);
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
  const [symbols, setSymbols] = useState<Symbol[]>([]);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [activeTab, setActiveTab] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [strategyParams, setStrategyParams] = useState<Record<string, any>>({});
  const [candleData, setCandleData] = useState([]);
  const [tradeData, setTradeData] = useState([]);

  // Timeframes don't change, so we can define them statically
  const timeframes = [
    { value: '1h', name: '1 Hour' },
    { value: '4h', name: '4 Hours' },
    { value: '1d', name: '1 Day' },
  ];

  // Handle timeframe selection (allow multiple)
  const handleTimeframeChange = (_event: React.MouseEvent<HTMLElement>, newTimeframes: string[]) => {
    if (newTimeframes.length > 0) {
      setSelectedTimeframes(newTimeframes);
    }
  };

  // Fetch symbols and strategies when component mounts
  useEffect(() => {
    async function fetchInitialData() {
      setIsLoading(true);
      try {
        // Fetch symbols
        const symbolsResponse = await apiService.getSymbols();
        if (symbolsResponse.status === 'success' && symbolsResponse.symbols) {
          const formattedSymbols = symbolsResponse.symbols.map((s: any) => ({
            value: s.symbol,
            name: `${s.baseAsset}/${s.quoteAsset}`
          }));
          setSymbols(formattedSymbols);
        }

        // Fetch strategies
        const strategiesResponse = await apiService.getStrategies();
        if (strategiesResponse.status === 'success' && strategiesResponse.strategies) {
          const formattedStrategies = strategiesResponse.strategies.map((s: any) => ({
            id: s.name,
            name: s.description,
            description: s.description,
            parameters: s.parameters
          }));
          setStrategies(formattedStrategies);
          
          // Set the first strategy as selected by default if available
          if (formattedStrategies.length > 0) {
            setSelectedStrategy(formattedStrategies[0].id);
            setStrategyParams(formattedStrategies[0].parameters);
          }
        }
      } catch (error) {
        console.error('Failed to fetch initial data:', error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchInitialData();
  }, []);

  // Update strategy parameters when strategy selection changes
  useEffect(() => {
    const selectedStrat = strategies.find(s => s.id === selectedStrategy);
    if (selectedStrat) {
      setStrategyParams(selectedStrat.parameters);
    }
  }, [selectedStrategy, strategies]);

  // Run the backtest
  const runBacktest = async () => {
    if (!selectedSymbol || selectedTimeframes.length === 0 || !selectedStrategy) {
      alert('Please select a symbol, at least one timeframe, and a strategy');
      return;
    }

    setIsRunning(true);
    setHasResults(false);
    
    try {
      const formattedStartDate = startDate ? startDate.toISOString().split('T')[0] : '';
      const formattedEndDate = endDate ? endDate.toISOString().split('T')[0] : '';
      
      if (!formattedStartDate || !formattedEndDate) {
        alert('Please select a valid date range');
        setIsRunning(false);
        return;
      }

      const backtestConfig = {
        symbol: selectedSymbol,
        timeframes: selectedTimeframes,
        start_date: formattedStartDate,
        end_date: formattedEndDate,
        initial_capital: initialCapital,
        commission: commission,
        strategy_name: selectedStrategy,
        strategy_params: strategyParams
      };

      const response = await apiService.runBacktest(backtestConfig);
      
      if (response.status === 'success') {
        setBacktestResult(response);
        setHasResults(true);
        
        // Format trade data for the chart
        if (response.trades && response.trades.length > 0) {
          const trades = response.trades.map((trade: any) => {
            const timestamp = new Date(trade.date).getTime() / 1000;
            return {
              time: timestamp,
              price: trade.entry_price,
              type: trade.side.toLowerCase() as 'buy' | 'sell'
            };
          });
          setTradeData(trades);
        }
        
        // Get market data for the chart
        try {
          const marketResponse = await apiService.getMarketData(
            selectedSymbol, 
            selectedTimeframes[0], 
            200
          );
          
          if (marketResponse.status === 'success' && marketResponse.candles) {
            const formattedCandles = marketResponse.candles
              .map((candle: any) => ({
                time: candle.time,
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
                // If there are duplicates, keep the latest one
                return acc;
              }
            }, []);
            
            setCandleData(uniqueCandles);
          }
        } catch (error) {
          console.error('Error fetching market data for chart:', error);
        }
      } else {
        alert('Backtest failed: ' + (response.message || 'Unknown error'));
      }
    } catch (error) {
      console.error('Error running backtest:', error);
      alert('Error running backtest. Please try again.');
    } finally {
      setIsRunning(false);
    }
  };

  // Handle parameter changes
  const handleParamChange = (param: string, value: any) => {
    setStrategyParams(prev => ({
      ...prev,
      [param]: value
    }));
  };

  // Handle tab change
  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  return (
    <ClientWrapper>
      <DashboardLayout>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" component="h1">Backtesting</Typography>
          <Button
            variant="contained"
            startIcon={<PlayArrowIcon />}
            onClick={runBacktest}
            disabled={isRunning || isLoading}
          >
            {isRunning ? "Running..." : "Run Backtest"}
          </Button>
        </Box>

        <Grid container spacing={3}>
          <Grid item xs={12} lg={3}>
            <Card sx={{ bgcolor: 'background.paper', mb: 3 }}>
              <CardHeader 
                title="Backtest Settings" 
                sx={{ bgcolor: 'action.selected', borderBottom: 1, borderColor: 'divider' }}
              />
              
              <CardContent>
                <Box component="div" sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" sx={{ mb: 1 }}>Market Settings</Typography>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel id="symbol-select-label">Symbol</InputLabel>
                    <Select
                      labelId="symbol-select-label"
                      value={selectedSymbol}
                      onChange={(e) => setSelectedSymbol(e.target.value)}
                      disabled={isLoading}
                      label="Symbol"
                      sx={{ bgcolor: 'background.paper' }}
                    >
                      {symbols.map((symbol) => (
                        <MenuItem key={symbol.value} value={symbol.value}>
                          {symbol.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" sx={{ mb: 1 }}>Timeframes</Typography>
                    <ToggleButtonGroup
                      value={selectedTimeframes}
                      onChange={handleTimeframeChange}
                      aria-label="timeframes"
                      disabled={isLoading}
                      size="small"
                      color="primary"
                      sx={{ display: 'flex', flexWrap: 'wrap' }}
                    >
                      {timeframes.map((timeframe) => (
                        <ToggleButton key={timeframe.value} value={timeframe.value}>
                          {timeframe.name}
                        </ToggleButton>
                      ))}
                    </ToggleButtonGroup>
                  </Box>
                </Box>
                
                <Divider sx={{ my: 2 }} />
                
                <Box component="div" sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" sx={{ mb: 1 }}>Strategy Settings</Typography>
                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <InputLabel id="strategy-select-label">Strategy</InputLabel>
                    <Select
                      labelId="strategy-select-label"
                      value={selectedStrategy}
                      onChange={(e: SelectChangeEvent) => setSelectedStrategy(e.target.value)}
                      disabled={isLoading}
                      label="Strategy"
                      sx={{ bgcolor: 'background.paper' }}
                    >
                      {strategies.map((strategy) => (
                        <MenuItem key={strategy.id} value={strategy.id}>
                          {strategy.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  {/* Show strategy parameters if a strategy is selected */}
                  {selectedStrategy && Object.keys(strategyParams).length > 0 && (
                    <Paper variant="outlined" sx={{ p: 2, mb: 2, bgcolor: 'background.default' }}>
                      <Typography variant="subtitle2" sx={{ mb: 1 }}>Strategy Parameters</Typography>
                      <FormGroup sx={{ gap: 2 }}>
                        {Object.entries(strategyParams).map(([param, value]) => (
                          <TextField
                            key={param}
                            label={param}
                            value={value}
                            onChange={(e) => handleParamChange(param, Number(e.target.value))}
                            type="number"
                            size="small"
                            inputProps={{
                              step: param.includes('period') ? 1 : 0.1,
                              min: 0,
                            }}
                            fullWidth
                          />
                        ))}
                      </FormGroup>
                    </Paper>
                  )}
                </Box>
                
                <Divider sx={{ my: 2 }} />
                
                <Box component="div" sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" sx={{ mb: 1 }}>Backtest Period</Typography>
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
                </Box>
                
                <Divider sx={{ my: 2 }} />
                
                <Box component="div" sx={{ mb: 3 }}>
                  <Typography variant="subtitle1" sx={{ mb: 1 }}>Account Settings</Typography>
                  <TextField
                    label="Initial Capital (USDT)"
                    value={initialCapital}
                    onChange={(e) => setInitialCapital(Number(e.target.value))}
                    type="number"
                    fullWidth
                    sx={{ mb: 2 }}
                    inputProps={{
                      min: 100,
                      max: 1000000,
                      step: 100,
                    }}
                    disabled={isLoading}
                  />
                  
                  <TextField
                    label="Commission (%)"
                    value={commission}
                    onChange={(e) => setCommission(Number(e.target.value))}
                    type="number"
                    fullWidth
                    inputProps={{
                      min: 0,
                      max: 5,
                      step: 0.01,
                    }}
                    disabled={isLoading}
                  />
                </Box>
                
                <Divider sx={{ my: 2 }} />
                
                <Button
                  variant="contained"
                  startIcon={<PlayArrowIcon />}
                  onClick={runBacktest}
                  disabled={isRunning || isLoading}
                  fullWidth
                  size="large"
                >
                  {isRunning ? "Running..." : "Run Backtest"}
                </Button>
              </CardContent>
            </Card>

            {hasResults && backtestResult && (
              <Card sx={{ bgcolor: 'background.paper' }}>
                <CardHeader 
                  title="Backtest Results" 
                  sx={{ bgcolor: 'action.selected', borderBottom: 1, borderColor: 'divider' }}
                />
                
                <CardContent>
                  <Box sx={{ '& > div': { py: 1, display: 'flex', justifyContent: 'space-between', borderBottom: 1, borderColor: 'divider' }}}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: 1, borderColor: 'divider' }}>
                      <Typography variant="body2" color="text.secondary">Net Profit</Typography>
                      <Typography variant="body2" color={backtestResult.metrics.total_profit >= 0 ? 'success.main' : 'error.main'} fontWeight="medium">
                        ${backtestResult.metrics.total_profit.toFixed(2)}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: 1, borderColor: 'divider' }}>
                      <Typography variant="body2" color="text.secondary">Return</Typography>
                      <Typography variant="body2" color={backtestResult.metrics.profit_percent >= 0 ? 'success.main' : 'error.main'} fontWeight="medium">
                        {backtestResult.metrics.profit_percent.toFixed(2)}%
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: 1, borderColor: 'divider' }}>
                      <Typography variant="body2" color="text.secondary">Max Drawdown</Typography>
                      <Typography variant="body2" color="error.main" fontWeight="medium">
                        {backtestResult.metrics.max_drawdown_percent.toFixed(2)}%
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: 1, borderColor: 'divider' }}>
                      <Typography variant="body2" color="text.secondary">Win Rate</Typography>
                      <Typography variant="body2" fontWeight="medium">
                        {(backtestResult.metrics.win_rate * 100).toFixed(2)}%
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: 1, borderColor: 'divider' }}>
                      <Typography variant="body2" color="text.secondary">Total Trades</Typography>
                      <Typography variant="body2" fontWeight="medium">
                        {backtestResult.metrics.total_trades}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1, borderBottom: 1, borderColor: 'divider' }}>
                      <Typography variant="body2" color="text.secondary">Winning Trades</Typography>
                      <Typography variant="body2" fontWeight="medium">
                        {backtestResult.metrics.winning_trades}
                      </Typography>
                    </Box>
                    
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', py: 1 }}>
                      <Typography variant="body2" color="text.secondary">Sharpe Ratio</Typography>
                      <Typography variant="body2" fontWeight="medium">
                        {backtestResult.metrics.sharpe_ratio.toFixed(2)}
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            )}
          </Grid>

          <Grid item xs={12} lg={9}>
            {hasResults && backtestResult ? (
              <Card sx={{ bgcolor: 'background.paper', height: '100%' }}>
                <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                  <Tabs value={activeTab} onChange={handleTabChange} aria-label="backtest results tabs">
                    <Tab label="Chart" />
                    <Tab label="Equity Curve" />
                    <Tab label="Trades" />
                  </Tabs>
                </Box>
                
                <CardContent>
                  <TabPanel value={activeTab} index={0}>
                    <Box sx={{ height: 500 }}>
                      {candleData.length > 0 ? (
                        <TradingChartDynamic 
                          candles={candleData}
                          trades={tradeData}
                          height={500}
                        />
                      ) : (
                        <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <Typography>No chart data available</Typography>
                        </Box>
                      )}
                    </Box>
                  </TabPanel>
                  
                  <TabPanel value={activeTab} index={1}>
                    <Box sx={{ height: 500 }}>
                      {backtestResult.equity_curve && backtestResult.equity_curve.length > 0 ? (
                        <LineChart
                          xAxis={[{
                            data: backtestResult.equity_curve.map(point => new Date(point.date)),
                            scaleType: 'time',
                          }]}
                          series={[{
                            data: backtestResult.equity_curve.map(point => point.balance),
                            label: 'Balance',
                            color: '#2196f3',
                          }]}
                          height={500}
                        />
                      ) : (
                        <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                          <Typography>No equity curve data available</Typography>
                        </Box>
                      )}
                    </Box>
                  </TabPanel>
                  
                  <TabPanel value={activeTab} index={2}>
                    <TableContainer component={Paper} sx={{ maxHeight: 500, overflow: 'auto' }}>
                      <Table stickyHeader>
                        <TableHead>
                          <TableRow>
                            <TableCell>Date</TableCell>
                            <TableCell>Symbol</TableCell>
                            <TableCell>Side</TableCell>
                            <TableCell>Entry Price</TableCell>
                            <TableCell>Exit Price</TableCell>
                            <TableCell>Quantity</TableCell>
                            <TableCell>Profit/Loss</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {backtestResult.trades && backtestResult.trades.length > 0 ? (
                            backtestResult.trades.map((trade, index) => (
                              <TableRow key={index}>
                                <TableCell>{trade.date}</TableCell>
                                <TableCell>{trade.symbol}</TableCell>
                                <TableCell sx={{ color: trade.side === 'BUY' ? 'success.main' : 'error.main' }}>
                                  {trade.side}
                                </TableCell>
                                <TableCell>${trade.entry_price.toFixed(2)}</TableCell>
                                <TableCell>${trade.exit_price.toFixed(2)}</TableCell>
                                <TableCell>{trade.quantity.toFixed(6)}</TableCell>
                                <TableCell sx={{ color: trade.profit_loss >= 0 ? 'success.main' : 'error.main' }}>
                                  ${trade.profit_loss.toFixed(2)}
                                </TableCell>
                              </TableRow>
                            ))
                          ) : (
                            <TableRow>
                              <TableCell colSpan={7} align="center">
                                No trade data available
                              </TableCell>
                            </TableRow>
                          )}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </TabPanel>
                </CardContent>
              </Card>
            ) : (
              <Card sx={{ bgcolor: 'background.paper', height: '100%' }}>
                <Box sx={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', p: 8 }}>
                  <Box sx={{ textAlign: 'center' }}>
                    <ShowChartIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
                    <Typography variant="h6" color="text.secondary" gutterBottom>
                      Run a backtest to see results
                    </Typography>
                    <Typography variant="body2" color="text.disabled">
                      Configure your backtest settings on the left panel and click "Run Backtest"
                    </Typography>
                  </Box>
                </Box>
              </Card>
            )}
          </Grid>
        </Grid>
      </DashboardLayout>
    </ClientWrapper>
  );
} 