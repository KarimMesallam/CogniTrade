'use client';

import { useState } from 'react';
import DashboardLayout from '../../components/DashboardLayout';
import ClientWrapper from '../../components/ClientWrapper';
import {
  Typography,
  Box,
  Card,
  CardContent,
  CardHeader,
  Button,
  TextField,
  Switch,
  Tabs,
  Tab,
  FormControlLabel,
  Grid,
  Divider,
  Paper,
  FormGroup,
  Chip
} from '@mui/material';

// Icons
import SaveIcon from '@mui/icons-material/Save';
import KeyIcon from '@mui/icons-material/Key';
import SettingsIcon from '@mui/icons-material/Settings';
import NotificationsIcon from '@mui/icons-material/Notifications';
import StorageIcon from '@mui/icons-material/Storage';
import SmartToyIcon from '@mui/icons-material/SmartToy';

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
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ py: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

export default function SettingsPage() {
  const [apiKey, setApiKey] = useState('18xo9p88s0plSSUIdtQ99n***********');
  const [apiSecret, setApiSecret] = useState('m28aUtiksT8giYSrZ63Q0m***********');
  const [isTestnet, setIsTestnet] = useState(true);
  const [defaultSymbol, setDefaultSymbol] = useState('BTCUSDT');
  
  const [llmApiKey, setLlmApiKey] = useState('');
  const [llmApiEndpoint, setLlmApiEndpoint] = useState('');
  const [llmModel, setLlmModel] = useState('');
  
  const [refreshInterval, setRefreshInterval] = useState(5);
  const [maxOrdersToKeep, setMaxOrdersToKeep] = useState(1000);
  const [maxSignalsToKeep, setMaxSignalsToKeep] = useState(1000);
  
  const [emailNotifications, setEmailNotifications] = useState(false);
  const [emailAddress, setEmailAddress] = useState('');
  const [telegramNotifications, setTelegramNotifications] = useState(false);
  const [telegramBotToken, setTelegramBotToken] = useState('');
  const [telegramChatId, setTelegramChatId] = useState('');

  const [showSecrets, setShowSecrets] = useState(false);
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setActiveTab(newValue);
  };

  const saveSettings = async () => {
    // Here you would typically save the settings to the backend
    console.log('Saving settings...');
  };

  return (
    <ClientWrapper>
      <DashboardLayout>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4" component="h1">Settings</Typography>
          <Button
            variant="contained"
            startIcon={<SaveIcon />}
            onClick={saveSettings}
          >
            Save Settings
          </Button>
        </Box>

        <Card sx={{ bgcolor: 'background.paper' }}>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs 
              value={activeTab} 
              onChange={handleTabChange} 
              variant="scrollable"
              scrollButtons="auto"
              aria-label="settings tabs"
            >
              <Tab icon={<KeyIcon />} label="API Credentials" iconPosition="start" />
              <Tab icon={<SmartToyIcon />} label="LLM Settings" iconPosition="start" />
              <Tab icon={<SettingsIcon />} label="General" iconPosition="start" />
              <Tab icon={<NotificationsIcon />} label="Notifications" iconPosition="start" />
              <Tab icon={<StorageIcon />} label="Database" iconPosition="start" />
            </Tabs>
          </Box>
          
          {/* API Credentials Tab */}
          <TabPanel value={activeTab} index={0}>
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardHeader title="Binance API Credentials" />
              <CardContent>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <TextField
                    label="API Key"
                    type={showSecrets ? "text" : "password"}
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="Enter your Binance API key"
                    fullWidth
                  />
                  
                  <TextField
                    label="API Secret"
                    type={showSecrets ? "text" : "password"}
                    value={apiSecret}
                    onChange={(e) => setApiSecret(e.target.value)}
                    placeholder="Enter your Binance API secret"
                    fullWidth
                  />
                  
                  <Divider />
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={showSecrets}
                        onChange={() => setShowSecrets(!showSecrets)}
                        color="primary"
                      />
                    }
                    label="Show secrets"
                  />
                  
                  <FormControlLabel
                    control={
                      <Switch
                        checked={isTestnet}
                        onChange={() => setIsTestnet(!isTestnet)}
                        color="primary"
                      />
                    }
                    label="Use Testnet (Paper Trading)"
                  />
                  
                  <TextField
                    label="Default Symbol"
                    value={defaultSymbol}
                    onChange={(e) => setDefaultSymbol(e.target.value)}
                    placeholder="BTCUSDT"
                    fullWidth
                  />
                </Box>
              </CardContent>
            </Card>
          </TabPanel>
          
          {/* LLM Settings Tab */}
          <TabPanel value={activeTab} index={1}>
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardHeader title="LLM Integration Settings" />
              <CardContent>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <TextField
                    label="LLM API Key"
                    type={showSecrets ? "text" : "password"}
                    value={llmApiKey}
                    onChange={(e) => setLlmApiKey(e.target.value)}
                    placeholder="Enter your LLM provider API key"
                    fullWidth
                  />
                  
                  <TextField
                    label="LLM API Endpoint"
                    value={llmApiEndpoint}
                    onChange={(e) => setLlmApiEndpoint(e.target.value)}
                    placeholder="https://api.yourllmprovider.com/v1"
                    fullWidth
                  />
                  
                  <TextField
                    label="LLM Model"
                    value={llmModel}
                    onChange={(e) => setLlmModel(e.target.value)}
                    placeholder="gpt-4o, claude-3-sonnet, etc."
                    fullWidth
                  />
                  
                  <Paper sx={{ p: 2, bgcolor: 'action.selected', borderRadius: 1 }}>
                    <Typography variant="subtitle2" gutterBottom>Note</Typography>
                    <Typography variant="body2">
                      LLM integration is optional. If provided, the trading bot will use the specified 
                      language model to enhance decision making. Without LLM credentials, the bot will 
                      fall back to rule-based decision making.
                    </Typography>
                  </Paper>
                </Box>
              </CardContent>
            </Card>
          </TabPanel>
          
          {/* General Settings Tab */}
          <TabPanel value={activeTab} index={2}>
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardHeader title="General Settings" />
              <CardContent>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <TextField
                    label="Data Refresh Interval (seconds)"
                    type="number"
                    value={refreshInterval}
                    onChange={(e) => setRefreshInterval(Number(e.target.value))}
                    inputProps={{ min: 1, max: 60, step: 1 }}
                    fullWidth
                  />
                  
                  <Paper sx={{ p: 2, bgcolor: 'action.selected', borderRadius: 1 }}>
                    <Typography variant="subtitle2" gutterBottom>Trading Mode</Typography>
                    <Grid container spacing={3}>
                      <Grid item xs={12} sm={6}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Box
                            sx={{
                              width: 12,
                              height: 12,
                              borderRadius: '50%',
                              bgcolor: isTestnet ? 'success.main' : 'grey.500'
                            }}
                          />
                          <Typography variant={isTestnet ? 'subtitle1' : 'body2'} color={isTestnet ? 'text.primary' : 'text.secondary'}>
                            Paper Trading
                          </Typography>
                        </Box>
                        <Typography variant="caption" color="text.secondary">
                          Safe mode using Binance Testnet
                        </Typography>
                      </Grid>
                      
                      <Grid item xs={12} sm={6}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Box
                            sx={{
                              width: 12,
                              height: 12,
                              borderRadius: '50%',
                              bgcolor: !isTestnet ? 'warning.main' : 'grey.500'
                            }}
                          />
                          <Typography variant={!isTestnet ? 'subtitle1' : 'body2'} color={!isTestnet ? 'text.primary' : 'text.secondary'}>
                            Live Trading
                          </Typography>
                        </Box>
                        <Typography variant="caption" color="text.secondary">
                          Real trading with actual funds
                        </Typography>
                      </Grid>
                    </Grid>
                  </Paper>
                </Box>
              </CardContent>
            </Card>
          </TabPanel>
          
          {/* Notifications Tab */}
          <TabPanel value={activeTab} index={3}>
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardHeader title="Notification Settings" />
              <CardContent>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <Paper sx={{ p: 2, bgcolor: 'action.selected', borderRadius: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="subtitle1">Email Notifications</Typography>
                      <Switch
                        checked={emailNotifications}
                        onChange={() => setEmailNotifications(!emailNotifications)}
                        color="primary"
                      />
                    </Box>
                    
                    {emailNotifications && (
                      <TextField
                        label="Email Address"
                        value={emailAddress}
                        onChange={(e) => setEmailAddress(e.target.value)}
                        placeholder="you@example.com"
                        fullWidth
                        size="small"
                        sx={{ mt: 2 }}
                      />
                    )}
                  </Paper>
                  
                  <Paper sx={{ p: 2, bgcolor: 'action.selected', borderRadius: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="subtitle1">Telegram Notifications</Typography>
                      <Switch
                        checked={telegramNotifications}
                        onChange={() => setTelegramNotifications(!telegramNotifications)}
                        color="primary"
                      />
                    </Box>
                    
                    {telegramNotifications && (
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
                        <TextField
                          label="Bot Token"
                          type={showSecrets ? "text" : "password"}
                          value={telegramBotToken}
                          onChange={(e) => setTelegramBotToken(e.target.value)}
                          placeholder="Telegram Bot Token"
                          fullWidth
                          size="small"
                        />
                        <TextField
                          label="Chat ID"
                          value={telegramChatId}
                          onChange={(e) => setTelegramChatId(e.target.value)}
                          placeholder="Telegram Chat ID"
                          fullWidth
                          size="small"
                        />
                      </Box>
                    )}
                  </Paper>
                </Box>
              </CardContent>
            </Card>
          </TabPanel>
          
          {/* Database Tab */}
          <TabPanel value={activeTab} index={4}>
            <Card sx={{ bgcolor: 'background.paper' }}>
              <CardHeader title="Database Settings" />
              <CardContent>
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <TextField
                    label="Maximum Orders to Keep"
                    type="number"
                    value={maxOrdersToKeep}
                    onChange={(e) => setMaxOrdersToKeep(Number(e.target.value))}
                    inputProps={{ min: 100, max: 10000, step: 100 }}
                    fullWidth
                    helperText="Orders older than this limit will be pruned to save space"
                  />
                  
                  <TextField
                    label="Maximum Signals to Keep"
                    type="number"
                    value={maxSignalsToKeep}
                    onChange={(e) => setMaxSignalsToKeep(Number(e.target.value))}
                    inputProps={{ min: 100, max: 10000, step: 100 }}
                    fullWidth
                    helperText="Signals older than this limit will be pruned to save space"
                  />
                  
                  <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                    <Button
                      variant="contained"
                      color="error"
                      size="small"
                    >
                      Clear Database
                    </Button>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </TabPanel>
        </Card>
      </DashboardLayout>
    </ClientWrapper>
  );
} 