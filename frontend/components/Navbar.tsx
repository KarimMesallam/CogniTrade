'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Divider,
  Chip
} from '@mui/material';

// MUI Icons
import HomeIcon from '@mui/icons-material/Home';
import BarChartIcon from '@mui/icons-material/BarChart';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import TimelineIcon from '@mui/icons-material/Timeline';
import HistoryIcon from '@mui/icons-material/History';
import SettingsIcon from '@mui/icons-material/Settings';
import CircleIcon from '@mui/icons-material/Circle';

interface NavItemProps {
  href: string;
  text: string;
  icon: React.ReactNode;
  isActive: boolean;
}

const drawerWidth = 240;

const NavItem = ({ href, text, icon, isActive }: NavItemProps) => {
  return (
    <ListItem disablePadding>
      <ListItemButton
        component={Link}
        href={href}
        selected={isActive}
        sx={{ 
          borderRadius: 1,
          mb: 0.5,
          '&.Mui-selected': {
            backgroundColor: 'primary.main',
            color: 'white',
            '&:hover': {
              backgroundColor: 'primary.dark',
            },
            '& .MuiListItemIcon-root': {
              color: 'white'
            }
          }
        }}
      >
        <ListItemIcon sx={{ color: isActive ? 'white' : 'text.secondary', minWidth: 36 }}>
          {icon}
        </ListItemIcon>
        <ListItemText primary={text} />
      </ListItemButton>
    </ListItem>
  );
};

export default function Navbar() {
  const pathname = usePathname();
  
  const navItems = [
    { href: '/', text: 'Home', icon: <HomeIcon fontSize="small" /> },
    { href: '/dashboard', text: 'Dashboard', icon: <BarChartIcon fontSize="small" /> },
    { href: '/trading', text: 'Live Trading', icon: <SmartToyIcon fontSize="small" /> },
    { href: '/backtesting', text: 'Backtesting', icon: <TimelineIcon fontSize="small" /> },
    { href: '/history', text: 'History', icon: <HistoryIcon fontSize="small" /> },
    { href: '/settings', text: 'Settings', icon: <SettingsIcon fontSize="small" /> },
  ];
  
  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        [`& .MuiDrawer-paper`]: { 
          width: drawerWidth, 
          boxSizing: 'border-box',
          bgcolor: 'background.default',
          borderRight: 1,
          borderColor: 'divider',
        },
      }}
    >
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
        <Box
          sx={{
            width: 32,
            height: 32,
            bgcolor: 'primary.main',
            borderRadius: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
        >
          <SmartToyIcon sx={{ color: 'white', fontSize: 20 }} />
        </Box>
        <Typography variant="h6" component="h1" fontWeight="bold">
          AI Trading Bot
        </Typography>
      </Box>
      
      <List sx={{ px: 2, mt: 2 }}>
        {navItems.map((item) => (
          <NavItem
            key={item.href}
            href={item.href}
            text={item.text}
            icon={item.icon}
            isActive={pathname === item.href}
          />
        ))}
      </List>
      
      <Box sx={{ position: 'absolute', bottom: 16, left: 0, right: 0, px: 2 }}>
        <Box
          sx={{
            bgcolor: 'action.selected',
            p: 2,
            borderRadius: 1,
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <CircleIcon sx={{ fontSize: 10, color: 'success.main' }} />
            <Typography variant="body2" color="text.secondary">
              API Connected
            </Typography>
          </Box>
          <Chip 
            label="Paper Trading Mode" 
            size="small" 
            variant="outlined" 
            color="primary"
            sx={{ fontSize: '0.7rem' }}
          />
        </Box>
      </Box>
    </Drawer>
  );
} 