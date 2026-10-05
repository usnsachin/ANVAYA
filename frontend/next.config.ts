import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Keep Next's static-generation process count low for constrained Windows CI.
  experimental: { cpus: 1 },
  async rewrites() {
    // The loopback default is development-only. Production must be configured
    // explicitly; Render uses NEXT_PUBLIC_API_URL for browser-to-API requests.
    const configuredOrigin = process.env.ANVAYA_BACKEND_ORIGIN?.trim();
    const backendOrigin = configuredOrigin || (
      process.env.NODE_ENV !== 'production' ? 'http://127.0.0.1:8000' : ''
    );
    if (!backendOrigin) return [];

    return [
      {
        source: '/api/:path*',
        destination: `${backendOrigin}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
