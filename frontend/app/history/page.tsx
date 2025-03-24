'use client';

import { useState, useEffect } from 'react';
import DashboardLayout from '../../components/DashboardLayout';
import ClientWrapper from '../../components/ClientWrapper';
import apiService from '../../lib/api-service';

// MUI Components
import {
  Typography,
  Box,
  Card,
  CardContent,
  Button,
  Tab,
  Tabs,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  SelectChangeEvent
} from '@mui/material';

// Icons
import SearchIcon from '@mui/icons-material/Search';
import DownloadIcon from '@mui/icons-material/Download';

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
  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
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
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" component="h1">Trading History</Typography>
          <Button
            variant="outlined"
            size="medium"
            startIcon={<DownloadIcon />}
            onClick={handleExportData}
            disabled={isLoading || (activeTab === 0 ? orders.length === 0 : signals.length === 0)}
          >
            Export Data
          </Button>
        </Box>

        <Card sx={{ bgcolor: 'background.paper', mb: 3 }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={activeTab} onChange={handleTabChange} aria-label="history tabs">
              <Tab label="Orders" />
              <Tab label="Signals" />
            </Tabs>
          </Box>
          
          <CardContent>
            <TabPanel value={activeTab} index={0}>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: 2, mb: 3 }}>
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  <FormControl sx={{ minWidth: 150 }}>
                    <InputLabel id="symbol-select-label">Symbol</InputLabel>
                    <Select
                      labelId="symbol-select-label"
                      value={selectedSymbol}
                      onChange={(e: SelectChangeEvent) => setSelectedSymbol(e.target.value)}
                      label="Symbol"
                      disabled={isLoading}
                    >
                      {symbols.map((symbol) => (
                        <MenuItem key={symbol.value} value={symbol.value}>
                          {symbol.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  <FormControl sx={{ minWidth: 100 }}>
                    <InputLabel id="limit-select-label">Items</InputLabel>
                    <Select
                      labelId="limit-select-label"
                      value={String(limit)}
                      onChange={(e: SelectChangeEvent) => setLimit(Number(e.target.value))}
                      label="Items"
                      disabled={isLoading}
                    >
                      <MenuItem value="20">20 items</MenuItem>
                      <MenuItem value="50">50 items</MenuItem>
                      <MenuItem value="100">100 items</MenuItem>
                      <MenuItem value="200">200 items</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
                
                <Button
                  variant="contained"
                  size="small"
                  startIcon={<SearchIcon />}
                  disabled={isLoading}
                >
                  Filter
                </Button>
              </Box>
              
              <TableContainer component={Paper} sx={{ maxHeight: 'calc(100vh - 300px)', overflow: 'auto' }}>
                <Table stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>Time</TableCell>
                      <TableCell>Symbol</TableCell>
                      <TableCell>Type</TableCell>
                      <TableCell>Price</TableCell>
                      <TableCell>Quantity</TableCell>
                      <TableCell>Value</TableCell>
                      <TableCell>Status</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {isLoading ? (
                      <TableRow>
                        <TableCell colSpan={7} align="center">
                          Loading...
                        </TableCell>
                      </TableRow>
                    ) : orders.length > 0 ? (
                      orders.map((order, index) => (
                        <TableRow key={index}>
                          <TableCell>{order.time}</TableCell>
                          <TableCell>{order.symbol}</TableCell>
                          <TableCell sx={{ color: order.type === 'BUY' ? 'success.main' : 'error.main' }}>
                            {order.type}
                          </TableCell>
                          <TableCell>
                            ${typeof order.price === 'number' ? order.price.toFixed(2) : order.price}
                          </TableCell>
                          <TableCell>{order.quantity}</TableCell>
                          <TableCell>
                            ${typeof order.value === 'number' ? order.value.toFixed(2) : order.value}
                          </TableCell>
                          <TableCell>
                            <Chip 
                              label={order.status}
                              size="small"
                              color={
                                order.status === 'FILLED' ? 'success' : 
                                order.status === 'PARTIAL' ? 'warning' : 
                                'error'
                              }
                              variant="outlined"
                            />
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={7} align="center">
                          No orders found
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </TabPanel>
            
            <TabPanel value={activeTab} index={1}>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', justifyContent: 'space-between', alignItems: 'center', gap: 2, mb: 3 }}>
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  <FormControl sx={{ minWidth: 150 }}>
                    <InputLabel id="symbol-select-label-signals">Symbol</InputLabel>
                    <Select
                      labelId="symbol-select-label-signals"
                      value={selectedSymbol}
                      onChange={(e: SelectChangeEvent) => setSelectedSymbol(e.target.value)}
                      label="Symbol"
                      disabled={isLoading}
                    >
                      {symbols.map((symbol) => (
                        <MenuItem key={symbol.value} value={symbol.value}>
                          {symbol.name}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                  
                  <FormControl sx={{ minWidth: 100 }}>
                    <InputLabel id="limit-select-label-signals">Items</InputLabel>
                    <Select
                      labelId="limit-select-label-signals"
                      value={String(limit)}
                      onChange={(e: SelectChangeEvent) => setLimit(Number(e.target.value))}
                      label="Items"
                      disabled={isLoading}
                    >
                      <MenuItem value="20">20 items</MenuItem>
                      <MenuItem value="50">50 items</MenuItem>
                      <MenuItem value="100">100 items</MenuItem>
                      <MenuItem value="200">200 items</MenuItem>
                    </Select>
                  </FormControl>
                </Box>
                
                <Button
                  variant="contained"
                  size="small"
                  startIcon={<SearchIcon />}
                  disabled={isLoading}
                >
                  Filter
                </Button>
              </Box>
              
              <TableContainer component={Paper} sx={{ maxHeight: 'calc(100vh - 300px)', overflow: 'auto' }}>
                <Table stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>Time</TableCell>
                      <TableCell>Symbol</TableCell>
                      <TableCell>Strategy</TableCell>
                      <TableCell>Signal</TableCell>
                      <TableCell>Confidence</TableCell>
                      <TableCell>Price</TableCell>
                      <TableCell>Action Taken</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {isLoading ? (
                      <TableRow>
                        <TableCell colSpan={7} align="center">
                          Loading...
                        </TableCell>
                      </TableRow>
                    ) : signals.length > 0 ? (
                      signals.map((signal, index) => (
                        <TableRow key={index}>
                          <TableCell>{signal.time}</TableCell>
                          <TableCell>{signal.symbol}</TableCell>
                          <TableCell>{signal.strategy}</TableCell>
                          <TableCell sx={{ color: signal.signal === 'BUY' ? 'success.main' : 'error.main' }}>
                            {signal.signal}
                          </TableCell>
                          <TableCell>
                            {typeof signal.confidence === 'number' ? signal.confidence.toFixed(2) : signal.confidence}
                          </TableCell>
                          <TableCell>
                            ${typeof signal.price === 'number' ? signal.price.toFixed(2) : signal.price}
                          </TableCell>
                          <TableCell sx={{ color: signal.action_taken.includes('Order') ? 'success.main' : 'text.secondary' }}>
                            {signal.action_taken}
                          </TableCell>
                        </TableRow>
                      ))
                    ) : (
                      <TableRow>
                        <TableCell colSpan={7} align="center">
                          No signals found
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </TableContainer>
            </TabPanel>
          </CardContent>
        </Card>
      </DashboardLayout>
    </ClientWrapper>
  );
} 