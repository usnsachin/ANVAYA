'use client';
import { Card, CardContent } from "@/components/ui/card";

interface ProbabilityCardProps {
  district: string;
  probability: number;
  threshold_mm: number;
}

export default function ProbabilityCard({ district, probability, threshold_mm }: ProbabilityCardProps) {
  const getProbabilityColor = (prob: number) => {
    if (prob >= 0.7) return 'bg-red-500';
    if (prob >= 0.4) return 'bg-amber-500';
    return 'bg-emerald-500';
  };

  return (
    <Card>
      <CardContent className="p-3 flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold">{district}</p>
          <p className="text-xs text-zinc-500">&gt; {threshold_mm}mm rain</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-bold">{(probability * 100).toFixed(0)}%</span>
          <div className={`${getProbabilityColor(probability)} w-3 h-3 rounded-full`} />
        </div>
      </CardContent>
    </Card>
  );
}
