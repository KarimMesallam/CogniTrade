'use client';

import { useState } from 'react';
import DashboardLayout from '../../components/DashboardLayout';
import { Title, Card, Text, Button, NumberInput, Switch, Tab, TabGroup, TabList, TabPanels, TabPanel, TextInput } from '@tremor/react';
import { FaSave, FaKey, FaCog, FaBell, FaExchangeAlt, FaRobot, FaDatabase } from 'react-icons/fa';

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

  const saveSettings = async () => {
    // Here you would typically save the settings to the backend
    console.log('Saving settings...');
  };

  return (
    <DashboardLayout>
      <div className="flex justify-between items-center mb-6">
        <Title>Settings</Title>
        <Button
          size="md"
          color="blue"
          onClick={saveSettings}
          icon={FaSave}
        >
          Save Settings
        </Button>
      </div>

      <div className="grid grid-cols-1 gap-6">
        <TabGroup>
          <TabList className="mb-4">
            <Tab icon={FaKey}>API Credentials</Tab>
            <Tab icon={FaRobot}>LLM Settings</Tab>
            <Tab icon={FaCog}>General</Tab>
            <Tab icon={FaBell}>Notifications</Tab>
            <Tab icon={FaDatabase}>Database</Tab>
          </TabList>
          
          <TabPanels>
            <TabPanel>
              <Card className="bg-slate-800 border-slate-700">
                <Title className="text-white mb-4">Binance API Credentials</Title>
                
                <div className="space-y-4">
                  <div>
                    <Text className="mb-2">API Key</Text>
                    <div className="flex gap-2">
                      <TextInput
                        type={showSecrets ? "text" : "password"}
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        placeholder="Enter your Binance API key"
                        className="flex-1"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <Text className="mb-2">API Secret</Text>
                    <div className="flex gap-2">
                      <TextInput
                        type={showSecrets ? "text" : "password"}
                        value={apiSecret}
                        onChange={(e) => setApiSecret(e.target.value)}
                        placeholder="Enter your Binance API secret"
                        className="flex-1"
                      />
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <Switch
                      id="show-secrets"
                      name="show-secrets"
                      checked={showSecrets}
                      onChange={() => setShowSecrets(!showSecrets)}
                    />
                    <Text>Show secrets</Text>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <Switch
                      id="testnet"
                      name="testnet"
                      checked={isTestnet}
                      onChange={() => setIsTestnet(!isTestnet)}
                    />
                    <Text>Use Testnet (Paper Trading)</Text>
                  </div>
                  
                  <div>
                    <Text className="mb-2">Default Symbol</Text>
                    <TextInput
                      value={defaultSymbol}
                      onChange={(e) => setDefaultSymbol(e.target.value)}
                      placeholder="BTCUSDT"
                    />
                  </div>
                </div>
              </Card>
            </TabPanel>
            
            <TabPanel>
              <Card className="bg-slate-800 border-slate-700">
                <Title className="text-white mb-4">LLM Integration Settings</Title>
                
                <div className="space-y-4">
                  <div>
                    <Text className="mb-2">LLM API Key</Text>
                    <TextInput
                      type={showSecrets ? "text" : "password"}
                      value={llmApiKey}
                      onChange={(e) => setLlmApiKey(e.target.value)}
                      placeholder="Enter your LLM provider API key"
                    />
                  </div>
                  
                  <div>
                    <Text className="mb-2">LLM API Endpoint</Text>
                    <TextInput
                      value={llmApiEndpoint}
                      onChange={(e) => setLlmApiEndpoint(e.target.value)}
                      placeholder="https://api.yourllmprovider.com/v1"
                    />
                  </div>
                  
                  <div>
                    <Text className="mb-2">LLM Model</Text>
                    <TextInput
                      value={llmModel}
                      onChange={(e) => setLlmModel(e.target.value)}
                      placeholder="gpt-4o, claude-3-sonnet, etc."
                    />
                  </div>
                  
                  <div className="p-3 bg-slate-700 rounded-md">
                    <Text className="text-white mb-2">Note</Text>
                    <Text className="text-gray-300 text-sm">
                      LLM integration is optional. If provided, the trading bot will use the specified 
                      language model to enhance decision making. Without LLM credentials, the bot will 
                      fall back to rule-based decision making.
                    </Text>
                  </div>
                </div>
              </Card>
            </TabPanel>
            
            <TabPanel>
              <Card className="bg-slate-800 border-slate-700">
                <Title className="text-white mb-4">General Settings</Title>
                
                <div className="space-y-4">
                  <div>
                    <Text className="mb-2">Data Refresh Interval (seconds)</Text>
                    <NumberInput
                      value={refreshInterval}
                      onValueChange={setRefreshInterval}
                      min={1}
                      max={60}
                      step={1}
                    />
                  </div>
                  
                  <div className="p-3 bg-slate-700 rounded-md">
                    <Text className="text-white mb-2">Trading Mode</Text>
                    <div className="flex items-center gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <div className={`h-3 w-3 rounded-full ${isTestnet ? 'bg-green-500' : 'bg-gray-500'}`}></div>
                          <Text className={isTestnet ? 'text-white' : 'text-gray-400'}>Paper Trading</Text>
                        </div>
                        <Text className="text-gray-400 text-xs mt-1">
                          Safe mode using Binance Testnet
                        </Text>
                      </div>
                      
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <div className={`h-3 w-3 rounded-full ${!isTestnet ? 'bg-amber-500' : 'bg-gray-500'}`}></div>
                          <Text className={!isTestnet ? 'text-white' : 'text-gray-400'}>Live Trading</Text>
                        </div>
                        <Text className="text-gray-400 text-xs mt-1">
                          Real trading with actual funds
                        </Text>
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            </TabPanel>
            
            <TabPanel>
              <Card className="bg-slate-800 border-slate-700">
                <Title className="text-white mb-4">Notification Settings</Title>
                
                <div className="space-y-4">
                  <div className="bg-slate-700 p-3 rounded-md">
                    <div className="flex items-center justify-between mb-2">
                      <Text className="text-white">Email Notifications</Text>
                      <Switch
                        id="email-notifications"
                        name="email-notifications"
                        checked={emailNotifications}
                        onChange={() => setEmailNotifications(!emailNotifications)}
                      />
                    </div>
                    
                    {emailNotifications && (
                      <div className="mt-2">
                        <Text className="mb-1">Email Address</Text>
                        <TextInput
                          value={emailAddress}
                          onChange={(e) => setEmailAddress(e.target.value)}
                          placeholder="you@example.com"
                        />
                      </div>
                    )}
                  </div>
                  
                  <div className="bg-slate-700 p-3 rounded-md">
                    <div className="flex items-center justify-between mb-2">
                      <Text className="text-white">Telegram Notifications</Text>
                      <Switch
                        id="telegram-notifications"
                        name="telegram-notifications"
                        checked={telegramNotifications}
                        onChange={() => setTelegramNotifications(!telegramNotifications)}
                      />
                    </div>
                    
                    {telegramNotifications && (
                      <div className="space-y-2 mt-2">
                        <div>
                          <Text className="mb-1">Bot Token</Text>
                          <TextInput
                            type={showSecrets ? "text" : "password"}
                            value={telegramBotToken}
                            onChange={(e) => setTelegramBotToken(e.target.value)}
                            placeholder="Telegram Bot Token"
                          />
                        </div>
                        <div>
                          <Text className="mb-1">Chat ID</Text>
                          <TextInput
                            value={telegramChatId}
                            onChange={(e) => setTelegramChatId(e.target.value)}
                            placeholder="Telegram Chat ID"
                          />
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </Card>
            </TabPanel>
            
            <TabPanel>
              <Card className="bg-slate-800 border-slate-700">
                <Title className="text-white mb-4">Database Settings</Title>
                
                <div className="space-y-4">
                  <div>
                    <Text className="mb-2">Maximum Orders to Keep</Text>
                    <NumberInput
                      value={maxOrdersToKeep}
                      onValueChange={setMaxOrdersToKeep}
                      min={100}
                      max={10000}
                      step={100}
                    />
                    <Text className="text-gray-400 text-xs mt-1">
                      Orders older than this limit will be pruned to save space
                    </Text>
                  </div>
                  
                  <div>
                    <Text className="mb-2">Maximum Signals to Keep</Text>
                    <NumberInput
                      value={maxSignalsToKeep}
                      onValueChange={setMaxSignalsToKeep}
                      min={100}
                      max={10000}
                      step={100}
                    />
                    <Text className="text-gray-400 text-xs mt-1">
                      Signals older than this limit will be pruned to save space
                    </Text>
                  </div>
                  
                  <div className="flex justify-end">
                    <Button
                      size="xs"
                      color="red"
                      className="mt-2"
                    >
                      Clear Database
                    </Button>
                  </div>
                </div>
              </Card>
            </TabPanel>
          </TabPanels>
        </TabGroup>
      </div>
    </DashboardLayout>
  );
} 