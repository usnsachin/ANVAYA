'use client';
import { Card, CardContent } from "@/components/ui/card";

interface RegimeCardProps {
  name: string;
  description: string;
  probability: number;
  color: string;
  isActive: boolean;
}

export default function RegimeCard({ name, description, probability, color, isActive }: RegimeCardProps) {
  return (
    <Card className={`relative overflow-hidden transition-all ${isActive ? 'ring-2 ring-zinc-900 dark:ring-zinc-100' : 'opacity-80'}`}>
      <div className="absolute top-0 left-0 w-1 h-full" style={{ backgroundColor: color }} />
      <CardContent className="p-4 pl-5">
        <div className="flex justify-between items-start mb-2">
          <h3 className="font-semibold text-sm">{name}</h3>
          <span className="text-xs font-bold bg-zinc-100 dark:bg-zinc-800 px-2 py-0.5 rounded">
            {(probability * 100).toFixed(0)}%
          </span>
        </div>
        <p className="text-xs text-zinc-500 dark:text-zinc-400 line-clamp-2">{description}</p>
      </CardContent>
    </Card>
  );
}
