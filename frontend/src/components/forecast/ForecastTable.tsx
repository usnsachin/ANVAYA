'use client';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import WhyForecast from "@/components/explainability/WhyForecast";

interface Forecast {
  district: string;
  date: string;
  raw_forecast_mm: number;
  corrected_forecast_mm: number;
  regime: string;
  confidence: number;
  very_heavy_probability?: number;
  extreme_probability?: number;
  prediction_mode?: string;
}

export default function ForecastTable({ forecasts }: { forecasts: Forecast[] }) {
  return (
    <div className="rounded-md border bg-white dark:bg-zinc-950 overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow className="bg-zinc-50 dark:bg-zinc-900/50 hover:bg-zinc-50 dark:hover:bg-zinc-900/50">
            <TableHead>District</TableHead>
            <TableHead>Date</TableHead>
            <TableHead className="text-right">Raw (mm)</TableHead>
            <TableHead className="text-right">Corrected (mm)</TableHead>
            <TableHead className="text-right">Heavy Rain Prob</TableHead>
            <TableHead className="text-right">VH / XH</TableHead>
            <TableHead>Regime</TableHead>
            <TableHead className="text-right"></TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {forecasts.map((f, i) => {
            return (
              <TableRow key={i}>
                <TableCell className="font-medium">{f.district}</TableCell>
                <TableCell className="text-zinc-500">{f.date}</TableCell>
                <TableCell className="text-right text-zinc-500 line-through decoration-zinc-300 dark:decoration-zinc-700">{f.raw_forecast_mm.toFixed(1)}</TableCell>
                <TableCell className="text-right font-medium">{f.corrected_forecast_mm.toFixed(1)}</TableCell>
                <TableCell className="text-right">
                  <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                    f.confidence > 0.8 ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400' :
                    f.confidence > 0.5 ? 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-400' :
                    'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-400'
                  }`}>
                    {(f.confidence * 100).toFixed(0)}%
                  </span>
                </TableCell>
                <TableCell className="text-right text-xs text-zinc-500">
                  {((f.very_heavy_probability || 0) * 100).toFixed(0)}% / {((f.extreme_probability || 0) * 100).toFixed(0)}%
                </TableCell>
                <TableCell>
                  <Badge variant="outline" className="text-[10px] uppercase tracking-wider">{f.regime}</Badge>
                </TableCell>
                <TableCell className="text-right">
                  <WhyForecast 
                    district={f.district} 
                    regime={f.regime} 
                    rawNwp={f.raw_forecast_mm} 
                    corrected={f.corrected_forecast_mm} 
                  />
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}
