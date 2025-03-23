'use client';

import { useState, useEffect } from 'react';
import DashboardLayout from '../../components/DashboardLayout';
import ClientWrapper from '../../components/ClientWrapper';
import { Title, Card, Text, Select, SelectItem, Button, DateRangePicker, DateRangePickerValue, NumberInput, Tab, TabGroup, TabList, TabPanels, TabPanel, LineChart } from '@tremor/react';
import { FaPlay, FaChartLine, FaCalendarAlt } from 'react-icons/fa';
import dynamic from 'next/dynamic';
import apiService from '../../lib/api-service';

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

export default function BacktestingPage() {
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  const [selectedTimeframes, setSelectedTimeframes] = useState(['1h']);
  const [selectedStrategy, setSelectedStrategy] = useState('');
  const [initialCapital, setInitialCapital] = useState(10000);
  const [commission, setCommission] = useState(0.1);
  const [dateRange, setDateRange] = useState<DateRangePickerValue>({
    from: new Date(new Date().getFullYear(), new Date().getMonth() - 3, 1), // 3 months ago
    to: new Date(),
  });
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
  const handleTimeframeChange = (timeframe: string) => {
    if (selectedTimeframes.includes(timeframe)) {
      setSelectedTimeframes(selectedTimeframes.filter(t => t !== timeframe));
    } else {
      setSelectedTimeframes([...selectedTimeframes, timeframe]);
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
      const startDate = dateRange.from ? dateRange.from.toISOString().split('T')[0] : '';
      const endDate = dateRange.to ? dateRange.to.toISOString().split('T')[0] : '';
      
      if (!startDate || !endDate) {
        alert('Please select a valid date range');
        setIsRunning(false);
        return;
      }

      const backtestConfig = {
        symbol: selectedSymbol,
        timeframes: selectedTimeframes,
        start_date: startDate,
        end_date: endDate,
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

  return (
    <ClientWrapper>
      <DashboardLayout>
        <div className="flex justify-between items-center mb-6">
          <Title>Backtesting</Title>
          <Button
            size="md"
            color="blue"
            onClick={runBacktest}
            icon={FaPlay}
            disabled={isRunning || isLoading}
          >
            {isRunning ? "Running..." : "Run Backtest"}
          </Button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-1">
            <Card className="bg-slate-800 border-slate-700 mb-6 overflow-hidden">
              <div className="border-b border-slate-700 px-4 py-3 bg-slate-700">
                <Title className="text-white text-lg">Backtest Settings</Title>
              </div>
              
              <div className="p-4 space-y-5">
                <div>
                  <Text className="text-white font-medium mb-2">Market Settings</Text>
                  <div className="space-y-3">
                    <div>
                      <Text className="mb-1 text-sm text-gray-300">Symbol</Text>
                      <Select
                        value={selectedSymbol}
                        onValueChange={setSelectedSymbol}
                        disabled={isLoading}
                        className="text-white border-slate-600 rounded-md px-3"
                      >
                        {symbols.map((symbol) => (
                          <SelectItem key={symbol.value} value={symbol.value} className="bg-slate-700 text-slate-200 hover:bg-blue-600">
                            {symbol.name}
                          </SelectItem>
                        ))}
                      </Select>
                    </div>
                    
                    <div>
                      <Text className="mb-1 text-sm text-gray-300">Timeframes</Text>
                      <div className="flex flex-wrap gap-2">
                        {timeframes.map((timeframe) => (
                          <button
                            key={timeframe.value}
                            onClick={() => handleTimeframeChange(timeframe.value)}
                            className={`px-3 py-1 text-sm rounded-md ${
                              selectedTimeframes.includes(timeframe.value)
                                ? 'bg-blue-600 text-white'
                                : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                            } transition-colors duration-200`}
                            disabled={isLoading}
                          >
                            {timeframe.name}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="border-t border-slate-700 pt-4">
                  <Text className="text-white font-medium mb-2">Strategy Settings</Text>
                  <div className="space-y-3">
                    <div>
                      <Text className="mb-1 text-sm text-gray-300">Strategy</Text>
                      <Select
                        value={selectedStrategy}
                        onValueChange={setSelectedStrategy}
                        disabled={isLoading}
                        className="text-white border-slate-600 rounded-md px-3"
                      >
                        {strategies.map((strategy) => (
                          <SelectItem key={strategy.id} value={strategy.id} className="bg-slate-700 text-slate-200 hover:bg-blue-600 w-full">
                            {strategy.name}
                          </SelectItem>
                        ))}
                      </Select>
                    </div>
                    
                    {/* Show strategy parameters if a strategy is selected */}
                    {selectedStrategy && Object.keys(strategyParams).length > 0 && (
                      <div className="bg-slate-700 p-3 rounded-md">
                        <Text className="text-sm font-medium text-white mb-2">Strategy Parameters</Text>
                        <div className="space-y-3">
                          {Object.entries(strategyParams).map(([param, value]) => (
                            <div key={param}>
                              <Text className="text-xs mb-1 text-gray-300">{param}</Text>
                              <NumberInput
                                value={value}
                                onValueChange={(val) => handleParamChange(param, val)}
                                min={0}
                                step={param.includes('period') ? 1 : 0.1}
                                className="px-3"
                              />
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="border-t border-slate-700 pt-4">
                  <Text className="text-white font-medium mb-2">Backtest Period</Text>
                  <div className="space-y-3">
                    <div>
                      <Text className="mb-1 text-sm text-gray-300">Date Range</Text>
                      <div className="relative">
                        <DateRangePicker
                          value={dateRange}
                          onValueChange={setDateRange}
                          className="w-full z-10 bg-slate-700 border-slate-600 text-white"
                          disabled={isLoading}
                          color="blue"
                        />
                        <style jsx global>{`
                          .tremor-DateRangePicker-root .tremor-DateRangePicker-calendarButton {
                            font-size: 16px;
                          }
                          .tremor-DateRangePicker-root .tremor-DateRangePicker-calendarButtonIcon {
                            width: 16px;
                            height: 16px;
                          }
                          .tremor-DateRangePicker-calendarModal {
                            z-index: 50;
                          }
                          .tremor-DateRangePicker-root .tremor-DateRangePicker-button {
                            padding-left: 12px;
                            padding-right: 12px;
                          }
                        `}</style>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="border-t border-slate-700 pt-4">
                  <Text className="text-white font-medium mb-2">Account Settings</Text>
                  <div className="space-y-3">
                    <div>
                      <Text className="mb-1 text-sm text-gray-300">Initial Capital (USDT)</Text>
                      <NumberInput
                        value={initialCapital}
                        onValueChange={setInitialCapital}
                        min={100}
                        max={1000000}
                        step={100}
                        disabled={isLoading}
                        className="px-3"
                      />
                    </div>
                    
                    <div>
                      <Text className="mb-1 text-sm text-gray-300">Commission (%)</Text>
                      <NumberInput
                        value={commission}
                        onValueChange={setCommission}
                        min={0}
                        max={5}
                        step={0.01}
                        disabled={isLoading}
                        className="px-3"
                      />
                    </div>
                  </div>
                </div>
                
                <div className="border-t border-slate-700 pt-4">
                  <Button
                    size="lg"
                    color="blue"
                    onClick={runBacktest}
                    icon={FaPlay}
                    disabled={isRunning || isLoading}
                    className="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-md shadow-md transition-colors duration-200"
                  >
                    {isRunning ? "Running..." : "Run Backtest"}
                  </Button>
                </div>
              </div>
            </Card>

            {hasResults && backtestResult && (
              <Card className="bg-slate-800 border-slate-700 overflow-hidden">
                <div className="border-b border-slate-700 px-4 py-3 bg-slate-700">
                  <Title className="text-white text-lg">Backtest Results</Title>
                </div>
                
                <div className="p-4 space-y-3">
                  <div className="flex justify-between py-1.5 border-b border-slate-700">
                    <Text className="text-gray-400">Net Profit</Text>
                    <Text className={`font-medium ${backtestResult.metrics.total_profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      ${backtestResult.metrics.total_profit.toFixed(2)}
                    </Text>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-700">
                    <Text className="text-gray-400">Return</Text>
                    <Text className={`font-medium ${backtestResult.metrics.profit_percent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {backtestResult.metrics.profit_percent.toFixed(2)}%
                    </Text>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-700">
                    <Text className="text-gray-400">Max Drawdown</Text>
                    <Text className="text-red-400 font-medium">
                      {backtestResult.metrics.max_drawdown_percent.toFixed(2)}%
                    </Text>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-700">
                    <Text className="text-gray-400">Win Rate</Text>
                    <Text className="text-white font-medium">
                      {(backtestResult.metrics.win_rate * 100).toFixed(2)}%
                    </Text>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-700">
                    <Text className="text-gray-400">Total Trades</Text>
                    <Text className="text-white font-medium">
                      {backtestResult.metrics.total_trades}
                    </Text>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-700">
                    <Text className="text-gray-400">Winning Trades</Text>
                    <Text className="text-white font-medium">
                      {backtestResult.metrics.winning_trades}
                    </Text>
                  </div>
                  <div className="flex justify-between py-1.5">
                    <Text className="text-gray-400">Sharpe Ratio</Text>
                    <Text className="text-white font-medium">
                      {backtestResult.metrics.sharpe_ratio.toFixed(2)}
                    </Text>
                  </div>
                </div>
              </Card>
            )}
          </div>

          {hasResults && backtestResult ? (
            <div className="lg:col-span-3">
              <Card className="bg-slate-800 border-slate-700 mb-6">
                <TabGroup>
                  <TabList className="mb-4">
                    <Tab>Chart</Tab>
                    <Tab>Equity Curve</Tab>
                    <Tab>Trades</Tab>
                  </TabList>
                  
                  <TabPanels>
                    <TabPanel>
                      <div className="h-[500px]">
                        {candleData.length > 0 ? (
                          <TradingChartDynamic 
                            candles={candleData}
                            trades={tradeData}
                            height={500}
                          />
                        ) : (
                          <div className="h-full flex items-center justify-center">
                            <Text>No chart data available</Text>
                          </div>
                        )}
                      </div>
                    </TabPanel>
                    
                    <TabPanel>
                      <div className="h-[500px]">
                        {backtestResult.equity_curve && backtestResult.equity_curve.length > 0 ? (
                          <LineChart
                            data={backtestResult.equity_curve}
                            index="date"
                            categories={["balance"]}
                            colors={["blue"]}
                            valueFormatter={(value) => `$${value.toFixed(2)}`}
                            yAxisWidth={60}
                            showAnimation
                          />
                        ) : (
                          <div className="h-full flex items-center justify-center">
                            <Text>No equity curve data available</Text>
                          </div>
                        )}
                      </div>
                    </TabPanel>
                    
                    <TabPanel>
                      <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-slate-700">
                          <thead>
                            <tr>
                              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Date</th>
                              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Symbol</th>
                              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Side</th>
                              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Entry Price</th>
                              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Exit Price</th>
                              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Quantity</th>
                              <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Profit/Loss</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-700">
                            {backtestResult.trades && backtestResult.trades.length > 0 ? (
                              backtestResult.trades.map((trade, index) => (
                                <tr key={index}>
                                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{trade.date}</td>
                                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{trade.symbol}</td>
                                  <td className={`px-4 py-3 whitespace-nowrap text-sm ${trade.side === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>
                                    {trade.side}
                                  </td>
                                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">${trade.entry_price.toFixed(2)}</td>
                                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">${trade.exit_price.toFixed(2)}</td>
                                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{trade.quantity.toFixed(6)}</td>
                                  <td className={`px-4 py-3 whitespace-nowrap text-sm ${trade.profit_loss >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                    ${trade.profit_loss.toFixed(2)}
                                  </td>
                                </tr>
                              ))
                            ) : (
                              <tr>
                                <td colSpan={7} className="px-4 py-8 text-center text-gray-400">
                                  No trade data available
                                </td>
                              </tr>
                            )}
                          </tbody>
                        </table>
                      </div>
                    </TabPanel>
                  </TabPanels>
                </TabGroup>
              </Card>
            </div>
          ) : (
            <div className="lg:col-span-3 flex items-center justify-center">
              <Card className="bg-slate-800 border-slate-700 w-full h-full flex items-center justify-center p-8">
                <div className="text-center">
                  <FaChartLine className="text-5xl text-gray-500 mx-auto mb-4" />
                  <Title className="text-gray-400 mb-2">Run a backtest to see results</Title>
                  <Text className="text-gray-500">
                    Configure your backtest settings on the left panel and click "Run Backtest"
                  </Text>
                </div>
              </Card>
            </div>
          )}
        </div>
      </DashboardLayout>
    </ClientWrapper>
  );
} 