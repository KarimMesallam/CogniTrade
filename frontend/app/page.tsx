'use client';

import DashboardLayout from '../components/DashboardLayout';
import { Card, Metric, Text, Title, BarList, Subtitle, AreaChart } from '@tremor/react';
import { useState } from 'react';

// Dashboard client component to prevent server errors
export default function Home() {
  return (
    <div className="p-8 bg-slate-900 min-h-screen text-white flex flex-col items-center justify-center">
      <div className="max-w-2xl text-center">
        <h1 className="text-4xl font-bold mb-6">AI Trading Bot Dashboard</h1>
        <p className="mb-8 text-lg">Web UI for AI Trading Bot with Binance & LLM Integration</p>
        
        <div className="flex gap-4 justify-center">
          <a href="/dashboard" className="px-6 py-3 bg-blue-600 rounded-md hover:bg-blue-700 transition-colors font-medium">
            Dashboard
          </a>
          <a href="/trading" className="px-6 py-3 bg-slate-700 rounded-md hover:bg-slate-600 transition-colors font-medium">
            Live Trading
          </a>
          <a href="/backtesting" className="px-6 py-3 bg-slate-700 rounded-md hover:bg-slate-600 transition-colors font-medium">
            Backtesting
          </a>
        </div>
      </div>
    </div>
  );
}

// Placeholder data for demonstration
const performanceData = [
  {
    date: 'Jan 22',
    'Profit/Loss': 2890,
  },
  {
    date: 'Feb 22',
    'Profit/Loss': 1890,
  },
  {
    date: 'Mar 22',
    'Profit/Loss': 3890,
  },
  {
    date: 'Apr 22',
    'Profit/Loss': -1290,
  },
  {
    date: 'May 22',
    'Profit/Loss': 1890,
  },
  {
    date: 'Jun 22',
    'Profit/Loss': 2890,
  },
];

const strategyPerformance = [
  { name: 'Moving Average Crossover', value: 421 },
  { name: 'RSI Strategy', value: 294 },
  { name: 'Bollinger Bands', value: -308 },
  { name: 'MACD Strategy', value: 152 },
];

// Client-only dashboard component
function ClientDashboard() {
  const [activeStrategy, setActiveStrategy] = useState('Moving Average Crossover');

  return (
    <DashboardLayout>
      <Title className="mb-4">Trading Dashboard</Title>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <Card className="bg-slate-800 border-slate-700">
          <Text className="text-gray-400">Total Profit/Loss</Text>
          <Metric className="text-white">$2,450.56</Metric>
        </Card>
        <Card className="bg-slate-800 border-slate-700">
          <Text className="text-gray-400">Win Rate</Text>
          <Metric className="text-white">68.2%</Metric>
        </Card>
        <Card className="bg-slate-800 border-slate-700">
          <Text className="text-gray-400">Total Trades</Text>
          <Metric className="text-white">42</Metric>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <Card className="bg-slate-800 border-slate-700">
          <Title className="text-white mb-2">Monthly Performance</Title>
          <AreaChart
            className="h-72 mt-4"
            data={performanceData}
            index="date"
            categories={['Profit/Loss']}
            colors={['blue']}
            showLegend={false}
            valueFormatter={(value: number) => `$${value}`}
            showAnimation={true}
          />
        </Card>

        <Card className="bg-slate-800 border-slate-700">
          <Title className="text-white mb-2">Strategy Performance</Title>
          <Subtitle className="text-gray-400 mb-4">
            Profit/Loss by strategy ($)
          </Subtitle>
          <BarList
            data={strategyPerformance}
            className="mt-2"
            valueFormatter={(value: number) => `$${value}`}
            color="blue"
          />
        </Card>
      </div>

      <div className="grid grid-cols-1 gap-6">
        <Card className="bg-slate-800 border-slate-700">
          <Title className="text-white mb-2">Recent Trades</Title>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-700">
              <thead>
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Time</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Symbol</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Type</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Price</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Quantity</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">P/L</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700">
                <tr>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">2023-03-23 14:22:05</td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">BTCUSDT</td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-green-400">BUY</td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">$42,560.78</td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">0.25</td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-green-400">+$125.45</td>
                </tr>
                <tr>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">2023-03-23 11:45:32</td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">ETHUSDT</td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-red-400">SELL</td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">$2,785.25</td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">1.5</td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-red-400">-$42.32</td>
                </tr>
                <tr>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">2023-03-22 18:33:17</td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">BTCUSDT</td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-green-400">BUY</td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">$42,105.92</td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">0.15</td>
                  <td className="px-4 py-3 whitespace-nowrap text-sm text-green-400">+$87.21</td>
                </tr>
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </DashboardLayout>
  );
}
