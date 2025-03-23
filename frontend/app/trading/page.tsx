'use client';

import React, { useEffect, useState } from 'react';
import DashboardLayout from '../../components/DashboardLayout';
import ClientWrapper from '../../components/ClientWrapper';
import { Title, Card, Text, Select, SelectItem, Button } from '@tremor/react';
import { FaPowerOff, FaPlay } from 'react-icons/fa';
import dynamic from 'next/dynamic';
import apiService from '../../lib/api-service';
import { useWebSocket } from '../../lib/websocket-context';

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
  
  const { isConnected, lastMessage, messages } = useWebSocket();

  // Handle WebSocket messages for real-time updates
  useEffect(() => {
    if (lastMessage) {
      if (lastMessage.type === 'trading_status') {
        setIsBotRunning(lastMessage.status === 'started');
      }
      else if (lastMessage.type === 'signal') {
        // Add the new signal to the signals array
        setSignals(prev => [lastMessage.data, ...prev].slice(0, 10));
      }
    }
  }, [lastMessage]);

  // Fetch market data when the symbol or timeframe changes
  useEffect(() => {
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
  }, [symbol, timeframe]);

  // Fetch signals when the symbol changes
  useEffect(() => {
    async function fetchSignals() {
      try {
        const response = await apiService.getSignalHistory(symbol, 10);
        if (response.status === 'success' && response.signals) {
          setSignals(response.signals);
        }
      } catch (error) {
        console.error('Failed to fetch signals:', error);
      }
    }

    fetchSignals();
  }, [symbol]);

  // Fetch symbols and strategies when the component mounts
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
            value: s.name,
            name: s.description
          }));
          setStrategies(formattedStrategies);
        }
      } catch (error) {
        console.error('Failed to fetch initial data:', error);
      } finally {
        setIsLoading(false);
      }
    }

    fetchInitialData();
  }, []);

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

  return (
    <ClientWrapper>
      <DashboardLayout>
        <div className="mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <Title>Live Trading</Title>
          <div className="flex flex-wrap gap-2">
            <Select 
              value={symbol} 
              onValueChange={setSymbol}
              className="w-44"
              disabled={isLoading}
            >
              {symbols.map((s) => (
                <SelectItem key={s.value} value={s.value}>
                  {s.name}
                </SelectItem>
              ))}
            </Select>
            <Select 
              value={timeframe} 
              onValueChange={setTimeframe}
              className="w-36"
            >
              {timeframes.map((t) => (
                <SelectItem key={t.value} value={t.value}>
                  {t.name}
                </SelectItem>
              ))}
            </Select>
            <Select 
              value={strategy} 
              onValueChange={setStrategy}
              className="w-52"
              disabled={isLoading}
            >
              {strategies.map((s) => (
                <SelectItem key={s.value} value={s.value}>
                  {s.name}
                </SelectItem>
              ))}
            </Select>
            <Button
              className={isBotRunning ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}
              onClick={toggleBot}
              icon={isBotRunning ? FaPowerOff : FaPlay}
              disabled={isLoading}
            >
              {isBotRunning ? 'Stop Bot' : 'Start Bot'}
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-6">
          <Card className="bg-slate-800 border-slate-700">
            <div className="flex justify-between items-center mb-4">
              <div className="flex items-center gap-2">
                <Text className="text-xl font-semibold text-white">{symbol}</Text>
                <Text className="text-sm text-gray-400">{timeframe}</Text>
              </div>
              <div className="flex items-center gap-2">
                <div className={`h-3 w-3 rounded-full ${isConnected ? (isBotRunning ? 'bg-green-500' : 'bg-yellow-500') : 'bg-red-500'}`}></div>
                <Text>{isConnected ? (isBotRunning ? 'Bot Running' : 'Bot Ready') : 'Disconnected'}</Text>
              </div>
            </div>
            {candleData.length > 0 ? (
              <TradingChart candles={candleData} trades={tradeData} height={400} />
            ) : (
              <div className="h-[400px] flex items-center justify-center">
                <Text>Loading chart data...</Text>
              </div>
            )}
          </Card>

          <Card className="bg-slate-800 border-slate-700">
            <Title className="text-white mb-2">Recent Signals</Title>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-slate-700">
                <thead>
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Time</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Symbol</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Strategy</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Signal</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Strength</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Action Taken</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700">
                  {signals.length > 0 ? (
                    signals.map((signal, index) => (
                      <tr key={index}>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{signal.time}</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{signal.symbol}</td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{signal.strategy}</td>
                        <td className={`px-4 py-3 whitespace-nowrap text-sm ${signal.signal === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>
                          {signal.signal}
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                          {typeof signal.confidence === 'number' ? signal.confidence.toFixed(2) : signal.confidence}
                        </td>
                        <td className={`px-4 py-3 whitespace-nowrap text-sm ${signal.action_taken.includes('Order') ? 'text-green-400' : 'text-gray-400'}`}>
                          {signal.action_taken}
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={6} className="px-4 py-8 text-center text-gray-400">
                        No signals available
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      </DashboardLayout>
    </ClientWrapper>
  );
} 