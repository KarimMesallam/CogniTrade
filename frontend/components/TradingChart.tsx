'use client';

import { useEffect, useRef } from 'react';
import { createChart, CandlestickSeries, LineSeries, ColorType } from 'lightweight-charts';

interface TradingChartProps {
  candles: {
    time: number;
    open: number;
    high: number;
    low: number;
    close: number;
    volume?: number;
  }[];
  trades?: {
    time: number;
    price: number;
    type: 'buy' | 'sell';
  }[];
  width?: number | string;
  height?: number | string;
  autosize?: boolean;
}

export default function TradingChart({ 
  candles, 
  trades = [], 
  width = '100%', 
  height = 500, 
  autosize = true 
}: TradingChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    try {
      // Initialize chart
      const chart = createChart(chartContainerRef.current, {
        width: autosize ? chartContainerRef.current.clientWidth : (typeof width === 'number' ? width : parseInt(width)),
        height: typeof height === 'number' ? height : parseInt(height),
        layout: {
          background: { type: ColorType.Solid, color: 'rgba(15, 23, 42, 1)' },
          textColor: 'rgba(255, 255, 255, 0.9)',
        },
        grid: {
          vertLines: { color: 'rgba(42, 46, 57, 0.5)' },
          horzLines: { color: 'rgba(42, 46, 57, 0.5)' },
        },
        timeScale: {
          timeVisible: true,
          secondsVisible: false,
          borderColor: 'rgba(197, 203, 206, 0.3)',
        },
        crosshair: {
          mode: 0,
        },
        rightPriceScale: {
          borderColor: 'rgba(197, 203, 206, 0.3)',
        },
      });

      // Create a map to store unique candles by timestamp
      const uniqueCandlesMap = new Map();

      // Process each candle and keep only the latest one for each timestamp
      if (candles && candles.length > 0) {
        candles.forEach(candle => {
          // Use timestamp as the key
          const timeKey = new Date(candle.time * 1000).toISOString().split('T')[0];
          uniqueCandlesMap.set(timeKey, {
            time: timeKey, // Format as YYYY-MM-DD
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
          });
        });
      }

      // Convert the map values to an array and sort it
      const formattedCandles = Array.from(uniqueCandlesMap.values())
        .sort((a, b) => {
          const dateA = new Date(a.time).getTime();
          const dateB = new Date(b.time).getTime();
          return dateA - dateB;
        });

      // Create candlestick series using the correct v5 API method
      const candleSeries = chart.addSeries(CandlestickSeries, {
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
      });

      // Set the data
      if (formattedCandles.length > 0) {
        candleSeries.setData(formattedCandles);
      }

      // Format trade data
      if (trades && trades.length > 0) {
        // Create a map for unique trade times
        const uniqueBuyTradesMap = new Map();
        const uniqueSellTradesMap = new Map();
        
        // Process trades
        trades.forEach(trade => {
          const timeKey = new Date(trade.time * 1000).toISOString().split('T')[0];
          if (trade.type === 'buy') {
            uniqueBuyTradesMap.set(timeKey, { 
              time: timeKey, 
              value: trade.price 
            });
          } else {
            uniqueSellTradesMap.set(timeKey, { 
              time: timeKey, 
              value: trade.price 
            });
          }
        });

        // Create buy marker series
        const buyMarkerSeries = chart.addSeries(LineSeries, {
          color: 'rgba(38, 166, 154, 1)',
          lineWidth: 2,
          lastValueVisible: false,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
        });

        // Create sell marker series
        const sellMarkerSeries = chart.addSeries(LineSeries, {
          color: 'rgba(239, 83, 80, 1)',
          lineWidth: 2,
          lastValueVisible: false,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
        });

        const buyTrades = Array.from(uniqueBuyTradesMap.values())
          .sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());
        
        const sellTrades = Array.from(uniqueSellTradesMap.values())
          .sort((a, b) => new Date(a.time).getTime() - new Date(b.time).getTime());

        if (buyTrades.length > 0) {
          buyMarkerSeries.setData(buyTrades);
        }

        if (sellTrades.length > 0) {
          sellMarkerSeries.setData(sellTrades);
        }
      }

      // Fit all data into the viewport
      chart.timeScale().fitContent();

      // Save chart reference
      chartRef.current = chart;

      // Resize handler
      const handleResize = () => {
        if (chartContainerRef.current && chartRef.current && autosize) {
          chartRef.current.applyOptions({ 
            width: chartContainerRef.current.clientWidth 
          });
        }
      };

      window.addEventListener('resize', handleResize);

      // Cleanup
      return () => {
        window.removeEventListener('resize', handleResize);
        chart.remove();
      };
    } catch (error) {
      console.error('Error creating chart:', error);
    }
  }, [candles, trades, width, height, autosize]);

  return (
    <div 
      ref={chartContainerRef} 
      style={{ 
        width: typeof width === 'string' ? width : `${width}px`, 
        height: typeof height === 'string' ? height : `${height}px` 
      }}
      className="tv-lightweight-charts bg-slate-900 rounded-md p-2"
    />
  );
} 