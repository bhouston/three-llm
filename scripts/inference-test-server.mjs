import { createServer } from 'vite';

// Serve the source kernels without the demo's SSR, telemetry, or model downloads.
const server = await createServer({
  configFile: false,
  optimizeDeps: { exclude: ['vgpu'] },
  plugins: [
    {
      name: 'inference-test-page',
      configureServer(vite) {
        vite.middlewares.use((req, res, next) => {
          if (req.url !== '/') return next();
          res.setHeader('Content-Type', 'text/html');
          res.end('<!doctype html><title>FP32 inference tests</title>');
        });
      },
    },
  ],
  server: { host: '127.0.0.1', port: 4178, strictPort: true },
});
await server.listen();
