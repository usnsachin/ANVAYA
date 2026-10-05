'use client';
import { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface Forecast {
  district: string;
  lat: number;
  lon: number;
  raw_forecast_mm: number;
  corrected_forecast_mm: number;
  confidence: number;
}

export default function RainfallMap({ forecasts = [] }: { forecasts?: Forecast[] }) {
  const [layer, setLayer] = useState('ai');

  // Bounding box for India roughly
  const minLon = 68.0;
  const maxLon = 97.0;
  const minLat = 8.0;
  const maxLat = 37.0;
  const lonRange = maxLon - minLon;
  const latRange = maxLat - minLat;

  return (
    <Card className="h-full flex flex-col w-full">
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle className="text-sm font-semibold uppercase tracking-wider text-zinc-500">Spatial Distribution</CardTitle>
          <Tabs value={layer} onValueChange={setLayer} className="w-[300px]">
            <TabsList className="grid w-full grid-cols-3 h-8">
              <TabsTrigger value="raw" className="text-[10px] px-1">Raw NWP</TabsTrigger>
              <TabsTrigger value="ai" className="text-[10px] px-1">AI Corrected</TabsTrigger>
              <TabsTrigger value="prob" className="text-[10px] px-1">Probability</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
      </CardHeader>
      <CardContent className="flex-1 p-4 pt-0">
        <div className="w-full h-full min-h-[350px] relative border border-zinc-200 dark:border-zinc-800 rounded-lg bg-[#f0f4f8] dark:bg-[#0f172a] overflow-hidden">
          
          {/* Faint grid lines to look like a map */}
          <div className="absolute inset-0" style={{ backgroundImage: 'linear-gradient(rgba(0,0,0,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.05) 1px, transparent 1px)', backgroundSize: '10% 10%' }}></div>
          
          <TooltipProvider>
            {forecasts.map((f, i) => {
              if (!f.lat || !f.lon) return null;
              
              const left = ((f.lon - minLon) / lonRange) * 100;
              const bottom = ((f.lat - minLat) / latRange) * 100;

              let value = 0;
              let color = 'bg-blue-500';
              if (layer === 'raw') {
                value = f.raw_forecast_mm;
                color = value > 50 ? 'bg-indigo-600' : value > 20 ? 'bg-blue-500' : 'bg-sky-400';
              } else if (layer === 'ai') {
                value = f.corrected_forecast_mm;
                color = value > 50 ? 'bg-indigo-600' : value > 20 ? 'bg-blue-500' : 'bg-sky-400';
              } else {
                value = f.confidence * 100;
                color = value > 70 ? 'bg-red-500' : value > 40 ? 'bg-amber-500' : 'bg-emerald-500';
              }
              
              // Size proportional to value for rainfall, fixed for probability
              const size = layer === 'prob' ? 14 : Math.max(8, Math.min(24, value / 3));

              return (
                <Tooltip key={i}>
                  <TooltipTrigger render={
                    <div 
                      className={`absolute rounded-full shadow-md transform -translate-x-1/2 translate-y-1/2 cursor-pointer border border-white dark:border-black transition-all duration-300 ${color}`}
                      style={{ 
                        left: `${left}%`, 
                        bottom: `${bottom}%`,
                        width: `${size}px`,
                        height: `${size}px`,
                        opacity: 0.8
                      }}
                    />
                  } />
                  <TooltipContent>
                    <div className="text-xs space-y-1 font-medium">
                      <p className="font-bold border-b pb-1">{f.district}</p>
                      <p>Raw: {f.raw_forecast_mm.toFixed(1)} mm</p>
                      <p>AI: {f.corrected_forecast_mm.toFixed(1)} mm</p>
                      <p>Heavy Prob: {(f.confidence * 100).toFixed(0)}%</p>
                    </div>
                  </TooltipContent>
                </Tooltip>
              );
            })}
          </TooltipProvider>

          <div className="absolute bottom-2 right-2 text-zinc-500 font-medium text-[10px] bg-white/80 dark:bg-black/80 px-2 py-1 rounded shadow-sm border backdrop-blur">
            {layer === 'raw' ? 'Raw NWP Rainfall' : layer === 'ai' ? 'AI Corrected Rainfall' : 'Heavy Rain Probability'}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
