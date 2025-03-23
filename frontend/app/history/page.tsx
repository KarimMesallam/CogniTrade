'use client';

import { useState, useEffect } from 'react';
import DashboardLayout from '../../components/DashboardLayout';
import ClientWrapper from '../../components/ClientWrapper';
import { Title, Card, Text, Tab, TabGroup, TabList, TabPanels, TabPanel, Select, SelectItem, Button } from '@tremor/react';
import { FaSearch, FaDownload } from 'react-icons/fa';
import apiService from '../../lib/api-service';

interface Order {
  time: string;
  symbol: string;
  type: string;
  price: number;
  quantity: number;
  value: number;
  status: string;
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

export default function HistoryPage() {
  const [selectedSymbol, setSelectedSymbol] = useState('all');
  const [limit, setLimit] = useState(50);
  const [orders, setOrders] = useState<Order[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [symbols, setSymbols] = useState([
    { value: 'all', name: 'All Symbols' },
  ]);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);

  // Fetch symbols when component mounts
  useEffect(() => {
    async function fetchSymbols() {
      try {
        const response = await apiService.getSymbols();
        if (response.status === 'success' && response.symbols) {
          // Create an array with 'All Symbols' as the first option
          const symbolOptions = [
            { value: 'all', name: 'All Symbols' },
            ...response.symbols.map((s: any) => ({ 
              value: s.symbol, 
              name: `${s.baseAsset}/${s.quoteAsset}` 
            }))
          ];
          setSymbols(symbolOptions);
        }
      } catch (error) {
        console.error('Error fetching symbols:', error);
      }
    }

    fetchSymbols();
  }, []);

  // Fetch orders based on selected symbol and limit
  useEffect(() => {
    async function fetchOrders() {
      setIsLoading(true);
      try {
        const symbolParam = selectedSymbol === 'all' ? undefined : selectedSymbol;
        const response = await apiService.getOrderHistory(symbolParam, limit);
        if (response.status === 'success' && response.orders) {
          setOrders(response.orders);
        }
      } catch (error) {
        console.error('Error fetching orders:', error);
      } finally {
        setIsLoading(false);
      }
    }

    if (activeTab === 0) {
      fetchOrders();
    }
  }, [selectedSymbol, limit, activeTab]);

  // Fetch signals based on selected symbol and limit
  useEffect(() => {
    async function fetchSignals() {
      setIsLoading(true);
      try {
        const symbolParam = selectedSymbol === 'all' ? undefined : selectedSymbol;
        const response = await apiService.getSignalHistory(symbolParam, limit);
        if (response.status === 'success' && response.signals) {
          setSignals(response.signals);
        }
      } catch (error) {
        console.error('Error fetching signals:', error);
      } finally {
        setIsLoading(false);
      }
    }

    if (activeTab === 1) {
      fetchSignals();
    }
  }, [selectedSymbol, limit, activeTab]);

  // Handle tab change
  const handleTabChange = (index: number) => {
    setActiveTab(index);
  };

  // Handle exporting data
  const handleExportData = () => {
    const data = activeTab === 0 ? orders : signals;
    if (data.length === 0) return;

    // Convert data to CSV format
    const headers = Object.keys(data[0]).join(',');
    const rows = data.map(item => Object.values(item).join(','));
    const csv = [headers, ...rows].join('\n');
    
    // Create a blob and download link
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('hidden', '');
    a.setAttribute('href', url);
    a.setAttribute('download', `${activeTab === 0 ? 'orders' : 'signals'}_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <ClientWrapper>
      <DashboardLayout>
        <div className="flex justify-between items-center mb-6">
          <Title>Trading History</Title>
          <Button
            size="sm"
            color="slate"
            icon={FaDownload}
            onClick={handleExportData}
            disabled={isLoading || (activeTab === 0 ? orders.length === 0 : signals.length === 0)}
          >
            Export Data
          </Button>
        </div>

        <Card className="bg-slate-800 border-slate-700 mb-6">
          <TabGroup onIndexChange={handleTabChange}>
            <TabList className="mb-4">
              <Tab>Orders</Tab>
              <Tab>Signals</Tab>
            </TabList>
            
            <TabPanels>
              <TabPanel>
                <div className="flex flex-wrap justify-between items-center gap-4 mb-4">
                  <div className="flex gap-4 flex-wrap">
                    <Select
                      value={selectedSymbol}
                      onValueChange={setSelectedSymbol}
                      className="w-40"
                      disabled={isLoading}
                    >
                      {symbols.map((symbol) => (
                        <SelectItem key={symbol.value} value={symbol.value}>
                          {symbol.name}
                        </SelectItem>
                      ))}
                    </Select>
                    
                    <Select
                      value={String(limit)}
                      onValueChange={(value) => setLimit(Number(value))}
                      className="w-32"
                      disabled={isLoading}
                    >
                      <SelectItem value="20">20 items</SelectItem>
                      <SelectItem value="50">50 items</SelectItem>
                      <SelectItem value="100">100 items</SelectItem>
                      <SelectItem value="200">200 items</SelectItem>
                    </Select>
                  </div>
                  
                  <div className="flex gap-2">
                    <Button
                      size="xs"
                      color="blue"
                      icon={FaSearch}
                      disabled={isLoading}
                    >
                      Filter
                    </Button>
                  </div>
                </div>
                
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-slate-700">
                    <thead>
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Time</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Symbol</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Type</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Price</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Quantity</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Value</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-700">
                      {isLoading ? (
                        <tr>
                          <td colSpan={7} className="px-4 py-8 text-center text-gray-400">
                            Loading...
                          </td>
                        </tr>
                      ) : orders.length > 0 ? (
                        orders.map((order, index) => (
                          <tr key={index}>
                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{order.time}</td>
                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{order.symbol}</td>
                            <td className={`px-4 py-3 whitespace-nowrap text-sm ${order.type === 'BUY' ? 'text-green-400' : 'text-red-400'}`}>
                              {order.type}
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                              ${typeof order.price === 'number' ? order.price.toFixed(2) : order.price}
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">{order.quantity}</td>
                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                              ${typeof order.value === 'number' ? order.value.toFixed(2) : order.value}
                            </td>
                            <td className="px-4 py-3 whitespace-nowrap text-sm">
                              <span className={`px-2 py-1 rounded-full text-xs ${
                                order.status === 'FILLED' ? 'bg-green-900 text-green-300' : 
                                order.status === 'PARTIAL' ? 'bg-amber-900 text-amber-300' : 
                                'bg-red-900 text-red-300'
                              }`}>
                                {order.status}
                              </span>
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={7} className="px-4 py-8 text-center text-gray-400">
                            No orders found
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </TabPanel>
              
              <TabPanel>
                <div className="flex flex-wrap justify-between items-center gap-4 mb-4">
                  <div className="flex gap-4 flex-wrap">
                    <Select
                      value={selectedSymbol}
                      onValueChange={setSelectedSymbol}
                      className="w-40"
                      disabled={isLoading}
                    >
                      {symbols.map((symbol) => (
                        <SelectItem key={symbol.value} value={symbol.value}>
                          {symbol.name}
                        </SelectItem>
                      ))}
                    </Select>
                    
                    <Select
                      value={String(limit)}
                      onValueChange={(value) => setLimit(Number(value))}
                      className="w-32"
                      disabled={isLoading}
                    >
                      <SelectItem value="20">20 items</SelectItem>
                      <SelectItem value="50">50 items</SelectItem>
                      <SelectItem value="100">100 items</SelectItem>
                      <SelectItem value="200">200 items</SelectItem>
                    </Select>
                  </div>
                  
                  <div className="flex gap-2">
                    <Button
                      size="xs"
                      color="blue"
                      icon={FaSearch}
                      disabled={isLoading}
                    >
                      Filter
                    </Button>
                  </div>
                </div>
                
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-slate-700">
                    <thead>
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Time</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Symbol</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Strategy</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Signal</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Confidence</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Price</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Action Taken</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-700">
                      {isLoading ? (
                        <tr>
                          <td colSpan={7} className="px-4 py-8 text-center text-gray-400">
                            Loading...
                          </td>
                        </tr>
                      ) : signals.length > 0 ? (
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
                            <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                              ${typeof signal.price === 'number' ? signal.price.toFixed(2) : signal.price}
                            </td>
                            <td className={`px-4 py-3 whitespace-nowrap text-sm ${
                              signal.action_taken.includes('Order') ? 'text-green-400' : 'text-gray-400'
                            }`}>
                              {signal.action_taken}
                            </td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={7} className="px-4 py-8 text-center text-gray-400">
                            No signals found
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
      </DashboardLayout>
    </ClientWrapper>
  );
} 