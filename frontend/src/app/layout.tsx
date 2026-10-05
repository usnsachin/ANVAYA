import type { Metadata } from "next";
import "./globals.css";
import "maplibre-gl/dist/maplibre-gl.css";
import Navbar from "@/components/layout/Navbar";
import { OperationalWorkspaceProvider } from "@/components/operational/OperationalWorkspace";

export const metadata: Metadata = {
  title: "ANVAYA | Rainfall Intelligence",
  description: "Location-focused rainfall guidance, district risk maps, and scientific forecast performance.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="min-h-screen flex flex-col bg-[#F5F8FA] text-[#12324A] antialiased">
        <Navbar />
        <main className="mx-auto flex min-w-0 w-full max-w-[1360px] flex-1 flex-col px-4 py-5 pb-24 sm:px-6 sm:py-7 md:pb-8 lg:px-8">
          <OperationalWorkspaceProvider>{children}</OperationalWorkspaceProvider>
        </main>
      </body>
    </html>
  );
}
