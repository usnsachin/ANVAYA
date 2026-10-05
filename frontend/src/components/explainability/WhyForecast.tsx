'use client';
import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerFooter,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer";
import { Button } from "@/components/ui/button";

interface WhyForecastProps {
  district: string;
  regime: string;
  rawNwp: number;
  corrected: number;
  trigger?: React.ReactElement;
}

export default function WhyForecast({ district, regime, rawNwp, corrected, trigger }: WhyForecastProps) {
  const adjustment = corrected - rawNwp;
  const isPositive = adjustment >= 0;

  return (
    <Drawer>
      <DrawerTrigger render={trigger || <Button variant="outline" size="sm" className="h-7 text-xs">Why this forecast?</Button>} />
      <DrawerContent>
        <div className="mx-auto w-full max-w-sm">
          <DrawerHeader>
            <DrawerTitle className="flex justify-between items-center">
              <span>Why Anvaya adjusted {district}</span>
              <span className="bg-amber-100 text-amber-800 text-[9px] font-bold px-1.5 py-0.5 rounded uppercase tracking-wider">Demo Data</span>
            </DrawerTitle>
            <DrawerDescription>AI-driven post-processing explanation</DrawerDescription>
          </DrawerHeader>
          <div className="p-4 pb-0 space-y-6">
            
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Detected Regime</h4>
              <p className="font-medium text-blue-600 dark:text-blue-400 capitalize">{regime} Monsoon</p>
            </div>

            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-zinc-500 uppercase tracking-wider">Key Signals</h4>
              <ul className="text-sm space-y-1 text-zinc-700 dark:text-zinc-300 list-disc pl-4">
                <li>Strong moisture transport</li>
                <li>High recent rainfall</li>
                <li>Elevated instability</li>
              </ul>
            </div>

            <div className="space-y-3 bg-zinc-50 dark:bg-zinc-900 p-4 rounded-lg border">
              <div className="flex justify-between items-center text-sm">
                <span className="text-zinc-500">Raw NWP</span>
                <span className="font-mono">{rawNwp.toFixed(1)} mm</span>
              </div>
              <div className="flex justify-between items-center text-sm font-medium">
                <span className="text-zinc-500">AI Adjustment</span>
                <span className={`font-mono ${isPositive ? 'text-emerald-600' : 'text-amber-600'}`}>
                  {isPositive ? '+' : ''}{adjustment.toFixed(1)} mm
                </span>
              </div>
              <div className="pt-2 mt-2 border-t flex justify-between items-center font-bold">
                <span>Corrected Forecast</span>
                <span className="font-mono text-lg">{corrected.toFixed(1)} mm</span>
              </div>
            </div>

          </div>
          <DrawerFooter>
            <DrawerClose render={<Button variant="outline">Close</Button>} />
          </DrawerFooter>
        </div>
      </DrawerContent>
    </Drawer>
  );
}
