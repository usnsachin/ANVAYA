'use client';
import { Card, CardContent } from "@/components/ui/card";

interface Forecast {
  district: string;
  raw_forecast_mm: number;
  corrected_forecast_mm: number;
  regime: string;
  confidence: number;
}

export default function RainfallSummary({ forecasts }: { forecasts: Forecast[] }) {
  const totalDistricts = forecasts.length;
  const avgCorrection = totalDistricts > 0 ? forecasts.reduce((acc, curr) => acc + (curr.corrected_forecast_mm - curr.raw_forecast_mm), 0) / totalDistricts : 0;
  const avgConfidence = totalDistricts > 0 ? forecasts.reduce((acc, curr) => acc + curr.confidence, 0) / totalDistricts : 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <Card>
        <CardContent className="p-4 flex flex-col justify-center">
          <p className="text-xs text-zinc-500 font-medium mb-1">Total Districts</p>
          <p className="text-2xl font-bold">{totalDistricts}</p>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 flex flex-col justify-center">
          <p className="text-xs text-zinc-500 font-medium mb-1">Avg Correction</p>
          <p className={`text-2xl font-bold ${avgCorrection > 0 ? 'text-blue-600' : avgCorrection < 0 ? 'text-red-600' : ''}`}>
            {avgCorrection > 0 ? '+' : ''}{avgCorrection.toFixed(1)} mm
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardContent className="p-4 flex flex-col justify-center">
          <p className="text-xs text-zinc-500 font-medium mb-1">Avg Confidence</p>
          <p className="text-2xl font-bold">{(avgConfidence * 100).toFixed(1)}%</p>
        </CardContent>
      </Card>
    </div>
  );
}
