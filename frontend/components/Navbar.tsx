'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { FaHome, FaChartLine, FaHistory, FaCog, FaRobot } from 'react-icons/fa';

interface NavItemProps {
  href: string;
  text: string;
  icon: React.ReactNode;
  isActive: boolean;
}

const NavItem = ({ href, text, icon, isActive }: NavItemProps) => {
  return (
    <Link 
      href={href} 
      className={`flex items-center gap-2 p-3 rounded-md transition-colors ${
        isActive 
          ? 'bg-blue-600 text-white' 
          : 'text-gray-400 hover:bg-slate-800 hover:text-white'
      }`}
    >
      <span className="text-lg">{icon}</span>
      <span className="font-medium">{text}</span>
    </Link>
  );
};

export default function Navbar() {
  const pathname = usePathname();
  
  const navItems = [
    { href: '/', text: 'Home', icon: <FaHome /> },
    { href: '/dashboard', text: 'Dashboard', icon: <FaChartLine /> },
    { href: '/trading', text: 'Live Trading', icon: <FaRobot /> },
    { href: '/backtesting', text: 'Backtesting', icon: <FaChartLine /> },
    { href: '/history', text: 'History', icon: <FaHistory /> },
    { href: '/settings', text: 'Settings', icon: <FaCog /> },
  ];
  
  return (
    <nav className="w-64 h-screen p-4 bg-slate-950 border-r border-slate-800 fixed">
      <div className="flex items-center gap-2 mb-8 px-3 pt-2">
        <div className="bg-blue-600 w-8 h-8 rounded-md flex items-center justify-center">
          <FaRobot className="text-white text-lg" />
        </div>
        <h1 className="text-xl font-bold text-white">AI Trading Bot</h1>
      </div>
      
      <div className="flex flex-col gap-1">
        {navItems.map((item) => (
          <NavItem
            key={item.href}
            href={item.href}
            text={item.text}
            icon={item.icon}
            isActive={pathname === item.href}
          />
        ))}
      </div>
      
      <div className="absolute bottom-8 left-0 right-0 px-4">
        <div className="bg-slate-800 p-3 rounded-md">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-3 h-3 rounded-full bg-green-500"></div>
            <span className="text-sm text-gray-300">API Connected</span>
          </div>
          <div className="text-xs text-gray-400">
            Paper Trading Mode
          </div>
        </div>
      </div>
    </nav>
  );
} 